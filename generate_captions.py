#!/usr/bin/env python3
"""Stage 1: generate K censored captions per image.

Example:
  python generate_captions.py --data-root data/cub/train --prompt prompts/cub_fine_grained.txt \
      --output captions/cub_train.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from cabm import captioning as C


def parse_args():
    p = argparse.ArgumentParser(description="CaBM caption generation")
    # data (choose one)
    p.add_argument("--data-root", help="Image folders: <root>/<class_name>/<image>")
    p.add_argument("--manifest", help="JSONL of {'image': path, 'label': class_name}")
    p.add_argument("--image-root", help="Prepended to relative manifest image paths")
    p.add_argument("--limit", type=int, default=0, help="Cap number of images (0 = all)")
    # captioner
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--prompt", required=True, help="Text file with the captioning prompt")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    # defaults: 5 captions, one per temperature
    p.add_argument("--temperatures", default="0.7,0.9,1.1,1.3,1.5")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--min-new-tokens", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=300)
    # censoring (class names removed by default)
    p.add_argument("--no-censor-class-names", action="store_true",
                   help="Do not remove class names from captions")
    p.add_argument("--censor-terms", help="JSON file with extra terms to censor "
                   "(e.g. group/family nouns); a list or {'terms': [...]}")
    p.add_argument("--replacement-token", default="subject")
    # output / sharding for multi-GPU runs
    p.add_argument("--output", required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, cls = self.samples[i]
        return f"{cls}/{Path(path).stem}", Image.open(path).convert("RGB"), C.clean_label(cls)


def main():
    args = parse_args()
    set_seed(args.seed)  
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()

    samples, class_names = C.build_samples(args.data_root, args.manifest, args.image_root)
    samples = [samples[i] for i in range(len(samples)) if i % args.num_shards == args.shard_id]
    if args.limit:
        samples = samples[:args.limit]

    # Censoring: class names (from the dataset labels) plus any user-provided terms.
    class_re = None if args.no_censor_class_names else C.class_name_regex(class_names)
    terms = C.load_censor_terms(args.censor_terms) if args.censor_terms else set()
    term_re = C.term_regex(terms)

    print(f"[captions] {len(samples)} images | {len(class_names)} classes | "
          f"K={len(temps)} temps={temps}")
    print(f"[censor] class_names={class_re is not None} extra_terms={len(terms)}")

    model, processor = C.load_captioner(args.model, args.dtype)
    decoding = dict(top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens)

    loader = DataLoader(ImageDataset(samples), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=lambda b: tuple(map(list, zip(*b))))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for ids, images, labels in tqdm(loader, desc="captioning"):
            per_temp = [C.caption_batch(model, processor, images, prompt, t, decoding) for t in temps]
            for b, (img_id, lab) in enumerate(zip(ids, labels)):
                row = {"image": img_id, "label": lab, "model": args.model}
                for k in range(len(temps)):
                    row[f"caption_{k + 1}"] = C.censor(
                        C.clean_text(per_temp[k][b]), class_re, term_re, args.replacement_token)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
    print(f"[done] wrote {written} rows -> {args.output}")


if __name__ == "__main__":
    main()
