"""Stage 2: text classifier over captions.

A pretrained transformer encoder is fine-tuned on the captions with a linear head
on the final-layer [CLS] representation. Image-level predictions
average the per-caption logits (logit voting). Data are split at the image level so
all K captions of an image stay in the same split.
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


def caption_keys(num_captions: int) -> List[str]:
    return [f"caption_{i}" for i in range(1, num_captions + 1)]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_records(paths: List[str]) -> List[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def build_label_mapping(records: List[dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({str(r["label"]) for r in records})
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def flatten_train(records, label2id, keys) -> Tuple[List[str], List[int], List[int]]:
    """Each caption becomes one labelled sample; also return the source image index."""
    texts, labels, image_idx = [], [], []
    for i, r in enumerate(records):
        lid = label2id[str(r["label"])]
        for k in keys:
            cap = r.get(k, "")
            if cap and cap.strip():
                texts.append(cap.strip())
                labels.append(lid)
                image_idx.append(i)
    return texts, labels, image_idx


def group_test(records, label2id, keys):
    """Per-caption test rows that keep image id and caption index for voting."""
    texts, labels, image_ids, cap_idx = [], [], [], []
    for r in records:
        lid = label2id[str(r["label"])]
        img = str(r.get("image", id(r)))
        for i, k in enumerate(keys, start=1):
            cap = r.get(k, "")
            if cap and cap.strip():
                texts.append(cap.strip())
                labels.append(lid)
                image_ids.append(img)
                cap_idx.append(i)
    return texts, labels, image_ids, cap_idx


def stratified_image_split(image_labels, val_ratio=0.1, seed=42):
    """Class-stratified split over image indices."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(image_labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio)) if len(idx) > 1 else 0
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# Model: encoder -> [CLS] -> linear head
# --------------------------------------------------------------------------- #
class CaptionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, label_smoothing=0.0, gradient_checkpointing=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return {"loss": loss, "logits": logits}


def load_trained(model_dir, device):
    """Rebuild a trained classifier from an output directory produced by stage 2."""
    import os
    from transformers import AutoTokenizer
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    lm = json.load(open(os.path.join(model_dir, "label_mapping.json")))
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    msub = os.path.join(model_dir, "model")
    ckpts = sorted((d for d in os.listdir(msub) if d.startswith("checkpoint-")),
                   key=lambda x: int(x.split("-")[-1]))
    ckpt = os.path.join(msub, ckpts[-1]) if ckpts else msub
    model = CaptionClassifier(cfg["model_name"], len(id2label))
    from safetensors.torch import load_file
    sd_path = os.path.join(ckpt, "model.safetensors")
    sd = load_file(sd_path) if os.path.exists(sd_path) else torch.load(
        os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.float().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    return model, tokenizer, id2label


# --------------------------------------------------------------------------- #
# Training extras: text augmentation, layer freezing, LLRD
# --------------------------------------------------------------------------- #
def augment_text(text, word_dropout, span_dropout):
    words = text.split()
    if span_dropout > 0 and len(words) > 20 and random.random() < span_dropout:
        span = random.randint(2, 4)
        start = random.randint(0, max(0, len(words) - span))
        words = words[:start] + words[start + span:]
    if word_dropout > 0:
        kept = [w for w in words if random.random() > word_dropout]
        if len(kept) > len(words) * 0.5:  # never drop more than half
            words = kept
    return " ".join(words)


class Collator:
    """Tokenize a batch; optionally apply light text augmentation (training only)."""
    def __init__(self, tokenizer, max_length, augment=False,
                 word_dropout=0.0, span_dropout=0.0, aug_prob=0.0):
        self.tok, self.max_length, self.augment = tokenizer, max_length, augment
        self.word_dropout, self.span_dropout, self.aug_prob = word_dropout, span_dropout, aug_prob

    def __call__(self, features):
        texts, labels = [], []
        for f in features:
            t = f["text"]
            if self.augment and self.aug_prob > 0 and random.random() < self.aug_prob:
                t = augment_text(t, self.word_dropout, self.span_dropout)
            texts.append(t)
            labels.append(int(f["labels"]))
        batch = self.tok(texts, truncation=True, max_length=self.max_length,
                         padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def freeze_lower_layers(model, k):
    if k <= 0:
        return
    for p in model.encoder.embeddings.parameters():
        p.requires_grad = False
    for layer in model.encoder.encoder.layer[:k]:
        for p in layer.parameters():
            p.requires_grad = False


def llrd_param_groups(model, base_lr, decay, weight_decay):
    """Layer-wise learning-rate decay: top layers get base_lr, lower layers less."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    n = len(model.encoder.encoder.layer)

    def lr_for(name):
        if name.startswith("classifier"):
            return base_lr
        if ".layer." in name:
            i = int(name.split(".layer.")[1].split(".")[0])
            return base_lr * (decay ** (n - 1 - i))
        if "embeddings" in name:
            return base_lr * (decay ** n)
        return base_lr

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (lr_for(name), 0.0 if any(x in name for x in no_decay) else weight_decay)
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


# --------------------------------------------------------------------------- #
# Metrics and logit voting
# --------------------------------------------------------------------------- #
def _topk(logits, labels, k):
    k = min(k, logits.shape[-1])
    top = np.argpartition(-logits, k - 1, axis=-1)[:, :k]
    return float((top == labels.reshape(-1, 1)).any(1).mean())


def _macro_f1(preds, labels):
    scores = []
    for c in np.unique(labels):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(eval_pred):
    logits, labels = (np.asarray(x) for x in eval_pred)
    preds = logits.argmax(-1)
    return {"accuracy": float((preds == labels).mean()),
            "top5": _topk(logits, labels, 5), "macro_f1": _macro_f1(preds, labels)}


def vote(logits, image_ids, cap_idx, labels, num_captions):
    """Aggregate per-caption logits into per-image predictions (average and max)."""
    by_image = defaultdict(lambda: {"logits": [], "label": None})
    for i, img in enumerate(image_ids):
        by_image[img]["logits"].append(logits[i])
        by_image[img]["label"] = labels[i]
    keys = sorted(by_image)
    per_caption = {}
    for ct in range(1, num_captions + 1):
        hit = tot = 0
        for i, img in enumerate(image_ids):
            if cap_idx[i] == ct:
                tot += 1
                hit += int(logits[i].argmax() == labels[i])
        if tot:
            per_caption[f"caption_{ct}_acc"] = hit / tot
    avg = np.stack([np.mean(by_image[k]["logits"], 0) for k in keys])
    mx = np.stack([np.max(by_image[k]["logits"], 0) for k in keys])
    y = np.asarray([by_image[k]["label"] for k in keys])
    res = dict(per_caption)
    res["avg_voting_acc"] = float((avg.argmax(1) == y).mean())
    res["avg_voting_top5"] = _topk(avg, y, 5)
    res["avg_voting_macro_f1"] = _macro_f1(avg.argmax(1), y)
    res["max_voting_acc"] = float((mx.argmax(1) == y).mean())
    res["n_images"] = int(len(keys))
    if per_caption:
        best = max(per_caption.values())
        res["best_single_caption_acc"] = best
        res["avg_gain_over_best_single"] = res["avg_voting_acc"] - best
    return res
