"""Stage 3: open-vocabulary concept discovery (paper Sec. 3.3).

Pipeline per class, over its correctly classified images:
  1. gradient x embedding token saliency -> word saliency
  2. propose candidate spans (variable-length n-grams, top-P, NMS)
  3. score each span by span-erasure (target-logit drop) -- the concept importance
  4. keep spans that matter on average and consistently
  5. cluster kept spans (HDBSCAN) into concepts and merge near-duplicate clusters
"""
import string
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Closed-class English function words (determiners, auxiliaries, prepositions,
# conjunctions, pronouns) used only for edge-trimming and filler rejection.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "and", "but", "if", "or", "because", "until",
    "while", "this", "that", "these", "those", "it", "its", "which",
    "what", "who", "whom", "whose", "their", "they", "them", "also", "sep",
}
# Boilerplate terms that VLMs use to narrate the image rather than the subject.
FILLERS = {
    "image", "picture", "photo", "photograph", "appears", "appear",
    "seen", "shows", "showing", "displayed", "captured",
    "background", "foreground", "seem", "seems", "look", "looks",
    "likely", "possibly", "probably", "perhaps", "suggesting",
    "overall", "single", "main", "central", "object", "figure",
}
EDGE_STRIP = STOPWORDS | FILLERS
_SUBWORD_STARTS = ("▁", "Ġ")  # SentencePiece '_', GPT-2 BPE 'G'


def _meaningful(word: str) -> bool:
    return len(word) > 1 and any(c.isalpha() for c in word)


def is_filler(phrase: str) -> bool:
    words = phrase.lower().split()
    return not words or all(w in EDGE_STRIP for w in words)


def normalize_phrase(phrase: str) -> str:
    words = phrase.lower().split()
    while len(words) > 1 and words[0] in EDGE_STRIP:
        words = words[1:]
    while len(words) > 1 and words[-1] in EDGE_STRIP:
        words = words[:-1]
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Token saliency (gradient x embedding)
# --------------------------------------------------------------------------- #
class Saliency:
    def __init__(self, model, tokenizer, device, max_length=512):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_length = max_length
        self.embed = model.encoder.embeddings.word_embeddings
        self.pad_id = tokenizer.pad_token_id

    def _tok(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        return {k: v.to(self.device) for k, v in enc.items()}, offsets

    @torch.no_grad()
    def logits(self, text) -> np.ndarray:
        enc, _ = self._tok(text)
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        return out["logits"][0].cpu().numpy()

    def word_saliency(self, text, target: int) -> Dict:
        enc, offsets = self._tok(text)
        input_ids, attn = enc["input_ids"], enc.get("attention_mask")
        store = {}

        def hook(_m, _i, out):
            store["emb"] = out
            out.retain_grad()
            return out

        h = self.embed.register_forward_hook(hook)
        try:
            self.model.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attn)["logits"]
            base = float(logits[0, target].detach().cpu())
            logits[0, target].backward()
            emb = store["emb"]
            sal = (emb.grad * emb).abs().sum(-1)[0].detach().cpu().numpy()
        finally:
            h.remove()

        sal[0] = 0.0                     # [CLS]
        if len(sal) > 1:
            sal[-1] = 0.0                # [SEP]
        if attn is not None:
            sal = sal * attn[0].detach().cpu().numpy()
        total = sal.sum()
        if total > 0:
            sal = sal / total            # normalize to sum to one

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        words, wsal, spans = self._to_words(text, tokens, sal, offsets)
        return {"words": words, "word_saliencies": wsal, "word_token_spans": spans,
                "input_ids": input_ids, "attention_mask": attn, "base_logit": base}

    @staticmethod
    def _to_words(text, tokens, sal, offsets):
        """Sum subword saliencies into word-level scores; return per-word token spans."""
        words, wsal, spans = [], [], []
        parts, buf, idx = [], [], []

        def flush():
            if not parts:
                return
            w = "".join(parts).strip(string.punctuation + string.whitespace)
            if w and _meaningful(w):
                words.append(w)
                wsal.append(float(sum(buf)))
                spans.append((idx[0], idx[-1] + 1))

        for ti, (tok, s, (a, b)) in enumerate(zip(tokens, sal, offsets)):
            if a == 0 and b == 0:        # special token
                continue
            if tok.startswith(_SUBWORD_STARTS) and parts:
                flush()
                parts, buf, idx = [], [], []
            piece = text[a:b]
            if tok.startswith(_SUBWORD_STARTS) and piece.startswith(" "):
                piece = piece[1:]
            parts.append(piece)
            buf.append(s)
            idx.append(ti)
        flush()
        return words, wsal, spans


# --------------------------------------------------------------------------- #
# Candidate proposal + erasure scoring
# --------------------------------------------------------------------------- #
def propose_phrases(words, saliencies, top_p=30, max_ngram=8) -> List[Tuple[str, float, int, int]]:
    """Top-P (phrase, score, word_start, word_end) via greedy NMS (<=50% overlap)."""
    total = sum(saliencies)
    if total <= 0:
        return []
    norm = [s / total for s in saliencies]
    cands = []
    for n in range(1, max_ngram + 1):
        for i in range(len(words) - n + 1):
            span = words[i:i + n]
            if not all(_meaningful(w) for w in span):
                continue
            raw = " ".join(w.lower() for w in span)
            phrase = normalize_phrase(raw)
            if not phrase or is_filler(phrase):
                continue
            trim = 0
            rw = raw.split()
            while trim < len(rw) and rw[trim] in EDGE_STRIP:
                trim += 1
            m = len(phrase.split())
            start = i + trim
            cands.append((phrase, float(sum(norm[start:start + m])), start, start + m))

    best = {}
    for phrase, score, s, e in cands:
        if phrase not in best or score > best[phrase][0]:
            best[phrase] = (score, s, e)
    ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)

    selected = []
    for phrase, (score, s, e) in ranked:
        span = set(range(s, e))
        redundant = False
        for sp, _, ss, se in selected:
            ov = len(span & set(range(ss, se)))
            if (ov and ov / min(len(span), se - ss) > 0.5) or phrase in sp or sp in phrase:
                redundant = True
                break
        if not redundant:
            selected.append((phrase, score, s, e))
        if len(selected) >= top_p:
            break
    return selected


@torch.no_grad()
def erasure_scores(model, input_ids, attention_mask, candidates, word_token_spans,
                   target, base_logit, pad_id) -> List[float]:
    """Clipped target-logit drop when each candidate span is dropped from attention."""
    if not candidates:
        return []
    n = len(candidates)
    ids = input_ids.repeat(n, 1)
    attn = attention_mask.repeat(n, 1) if attention_mask is not None else torch.ones_like(ids)
    for b, (_, _, ws, we) in enumerate(candidates):
        t0 = word_token_spans[ws][0]
        t1 = word_token_spans[we - 1][1]
        ids[b, t0:t1] = pad_id
        attn[b, t0:t1] = 0
    masked = model(input_ids=ids, attention_mask=attn)["logits"][:, target].detach().cpu().numpy()
    return [max(0.0, float(base_logit - m)) for m in masked]


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_phrases(phrases, model, tokenizer, device, batch_size=256, max_length=64) -> np.ndarray:
    """L2-normalized [CLS] embeddings from the trained encoder."""
    out = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model.encoder(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        cls = F.normalize(h.last_hidden_state[:, 0], p=2, dim=-1)
        out.append(cls.cpu().numpy())
    return np.concatenate(out, 0)


def cluster(embeddings, min_cluster_size=5, min_samples=3) -> np.ndarray:
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           metric="euclidean", cluster_selection_method="eom").fit_predict(embeddings)


def merge_by_centroid(embeddings, labels, threshold=0.70) -> np.ndarray:
    """Merge clusters whose L2-normalized centroids have cosine similarity >= threshold."""
    unique = sorted(set(labels.tolist()) - {-1})
    if len(unique) < 2:
        return labels
    cent = {}
    for c in unique:
        v = embeddings[labels == c].mean(0)
        cent[c] = v / (np.linalg.norm(v) + 1e-9)
    parent = {c: c for c in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            a, b = unique[i], unique[j]
            if float(np.dot(cent[a], cent[b])) >= threshold:
                parent[find(b)] = find(a)
    remap, nid, new = {}, 0, labels.copy()
    for c in unique:
        r = find(c)
        if r not in remap:
            remap[r] = nid
            nid += 1
    for c in unique:
        new[labels == c] = remap[find(c)]
    return new


def summarize(phrases, embeddings, labels, scores, top_concepts=50) -> List[dict]:
    """One concept per cluster: the centroid-closest phrase and summed erasure importance."""
    clusters = []
    for c in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == c)[0]
        emb = embeddings[idx]
        centroid = emb.mean(0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        rep = idx[np.dot(emb, centroid).argmax()]
        clusters.append({
            "concept": phrases[rep],
            "importance": round(float(sum(scores[i] for i in idx)), 4),
        })
    clusters.sort(key=lambda c: c["importance"], reverse=True)
    return clusters[:top_concepts]
