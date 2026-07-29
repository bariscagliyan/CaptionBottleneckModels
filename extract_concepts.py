#!/usr/bin/env python3
"""Stage 3: extract per-class concepts with importance scores.

Example:
  python extract_concepts.py --captions captions/cub_train.jsonl \
      --model-dir runs/cub --output-dir runs/cub/concepts

Output (under --output-dir):
  final_concepts.json   class -> [{concept, importance}]  (one entry per cluster,
                        the centroid-closest phrase and its summed erasure importance)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from cabm import classifier as M
from cabm import concepts as X


def parse_args():
    p = argparse.ArgumentParser(description="CaBM concept extraction")
    p.add_argument("--captions", nargs="+", required=True, help="Training captions (from stage 1)")
    p.add_argument("--model-dir", required=True, help="Trained classifier dir (from stage 2)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-captions", type=int, default=5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-images-per-class", type=int, default=0, help="0 = all correct images")
    # proposal
    p.add_argument("--top-phrases-per-caption", type=int, default=30)
    p.add_argument("--max-ngram", type=int, default=8)
    # retention
    p.add_argument("--min-mean-delta", type=float, default=0.1)
    p.add_argument("--min-positive-rate", type=float, default=0.60)
    # clustering
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--merge-threshold", type=float, default=0.70)
    p.add_argument("--top-concepts", type=int, default=50)
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, id2label = M.load_trained(args.model_dir, device)
    label2id = {v: k for k, v in id2label.items()}
    keys = M.caption_keys(args.num_captions)
    records = M.load_records(args.captions)
    sal = X.Saliency(model, tokenizer, device, args.max_length)
    pad_id = tokenizer.pad_token_id

    # Stage 1: propose + erasure-score spans over correctly classified images.
    print("[1/2] saliency proposal + erasure scoring")
    class_deltas = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    n_total = n_correct = 0
    for r in tqdm(records, desc="images"):
        label = str(r["label"])
        if label not in label2id:
            continue
        lid = label2id[label]
        if args.max_images_per_class and counts[lid] >= args.max_images_per_class:
            continue
        caps = [r[k].strip() for k in keys if r.get(k) and r[k].strip()]
        if not caps:
            continue
        n_total += 1
        avg_logits = np.mean([sal.logits(c) for c in caps], axis=0)
        if int(avg_logits.argmax()) != lid:          # correctness filter
            continue
        n_correct += 1
        used = False
        for cap in caps:
            s = sal.word_saliency(cap, lid)
            if not s["words"]:
                continue
            cands = X.propose_phrases(s["words"], s["word_saliencies"],
                                      args.top_phrases_per_caption, args.max_ngram)
            if not cands:
                continue
            deltas = X.erasure_scores(model, s["input_ids"], s["attention_mask"], cands,
                                      s["word_token_spans"], lid, s["base_logit"], pad_id)
            for (phrase, _, _, _), d in zip(cands, deltas):
                p = X.normalize_phrase(phrase)
                if p and not X.is_filler(p):
                    class_deltas[lid][p].append(d)
                    used = True
        if used:
            counts[lid] += 1
    print(f"  images={n_total} correct={n_correct}")

    # Stage 2: retain phrases, then cluster into final concepts.
    print("[2/2] retention + clustering")
    final = {}
    for lid in sorted(id2label):
        name = id2label[lid]
        kept = []
        for phrase, deltas in class_deltas.get(lid, {}).items():
            if (np.mean(deltas) >= args.min_mean_delta
                    and np.mean([d > 0 for d in deltas]) >= args.min_positive_rate):
                kept.append((phrase, float(np.sum(deltas))))
        if len(kept) < args.min_cluster_size:
            final[name] = []
            continue
        kept.sort(key=lambda x: x[1], reverse=True)
        phrases = [p for p, _ in kept]
        scores = [s for _, s in kept]
        emb = X.embed_phrases(phrases, model, tokenizer, device, args.embed_batch_size)
        labels = X.cluster(emb, args.min_cluster_size, args.min_samples)
        labels = X.merge_by_centroid(emb, labels, args.merge_threshold)
        final[name] = X.summarize(phrases, emb, labels, scores, args.top_concepts)

    out = os.path.join(args.output_dir, "final_concepts.json")
    with open(out, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"[done] {sum(len(v) for v in final.values())} concepts across "
          f"{sum(1 for v in final.values() if v)} classes -> {out}")


if __name__ == "__main__":
    main()
