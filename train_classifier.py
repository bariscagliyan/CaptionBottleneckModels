#!/usr/bin/env python3
"""Stage 2: train the caption text classifier.

Example:
  python train_classifier.py --train-captions captions/cub_train.jsonl \
      --test-captions captions/cub_test.jsonl --output-dir runs/cub
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

from cabm import classifier as M


def parse_args():
    p = argparse.ArgumentParser(description="CaBM text classifier training")
    p.add_argument("--train-captions", nargs="+", required=True)
    p.add_argument("--test-captions", nargs="+", help="Optional; evaluated with logit voting")
    p.add_argument("--val-captions", nargs="+",
                   help="Explicit validation captions; if omitted, hold out --val-split from train")
    p.add_argument("--val-split", type=float, default=0.1, help="Val fraction when --val-captions unset")
    p.add_argument("--num-captions", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="roberta-base")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=3)  # effective batch size 48
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--llrd-factor", type=float, default=0.85, help="0 disables LLRD")
    p.add_argument("--freeze-layers", type=int, default=2)
    p.add_argument("--word-dropout", type=float, default=0.1)
    p.add_argument("--span-dropout", type=float, default=0.03)
    p.add_argument("--aug-prob", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class CaptionTrainer(Trainer):
    """Trainer with an augmentation-free eval collator and optional LLRD optimizer."""
    def __init__(self, *a, eval_collator=None, llrd_factor=0.0, **kw):
        super().__init__(*a, **kw)
        self.eval_collator, self.llrd_factor = eval_collator, llrd_factor

    def _swap(self, fn, *a):
        orig = self.data_collator
        self.data_collator = self.eval_collator or orig
        try:
            return fn(*a)
        finally:
            self.data_collator = orig

    def get_eval_dataloader(self, eval_dataset=None):
        return self._swap(super().get_eval_dataloader, eval_dataset)

    def get_test_dataloader(self, test_dataset):
        return self._swap(super().get_test_dataloader, test_dataset)

    def create_optimizer(self):
        if not self.llrd_factor:
            return super().create_optimizer()
        if self.optimizer is None:
            groups = M.llrd_param_groups(self.model, self.args.learning_rate,
                                         self.llrd_factor, self.args.weight_decay)
            self.optimizer = torch.optim.AdamW(groups, lr=self.args.learning_rate)
        return self.optimizer


def resolve_precision(mode):
    if not torch.cuda.is_available():
        return False, False
    if mode == "fp16":
        return True, False
    if mode == "bf16":
        return False, True
    if mode == "fp32":
        return False, False
    return (False, True) if torch.cuda.is_bf16_supported() else (True, False)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = M.caption_keys(args.num_captions)

    train_records = M.load_records(args.train_captions)
    label2id, id2label = M.build_label_mapping(train_records)
    texts, labels, image_idx = M.flatten_train(train_records, label2id, keys)

    if args.val_captions:
        # Use an explicit validation split (its labels must be within the train label set).
        tr = list(range(len(texts)))
        val_records = M.load_records(args.val_captions)
        val_records = [r for r in val_records if str(r["label"]) in label2id]
        v_texts, v_labels, _ = M.flatten_train(val_records, label2id, keys)
        print(f"[data] classes={len(id2label)} train_caps={len(tr)} "
              f"val_caps={len(v_texts)} (explicit --val-captions)")
    else:
        # Hold out a class-stratified fraction of the training images.
        image_labels = [label2id[str(r["label"])] for r in train_records]
        tr_img, val_img = M.stratified_image_split(image_labels, args.val_split, args.seed)
        tr_img, val_img = set(tr_img), set(val_img)
        tr = [i for i, im in enumerate(image_idx) if im in tr_img]
        va = [i for i, im in enumerate(image_idx) if im in val_img]
        v_texts = [texts[i] for i in va]
        v_labels = [labels[i] for i in va]
        print(f"[data] classes={len(id2label)} train_caps={len(tr)} "
              f"val_caps={len(va)} (val_split={args.val_split})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = Dataset.from_dict({"text": [texts[i] for i in tr], "labels": [labels[i] for i in tr]})
    val_ds = Dataset.from_dict({"text": v_texts, "labels": v_labels})

    model = M.CaptionClassifier(args.model_name, len(id2label),
                                label_smoothing=args.label_smoothing)
    model.float()  # some encoder checkpoints load as fp16; keep fp32 master weights
    M.freeze_lower_layers(model, args.freeze_layers)

    fp16, bf16 = resolve_precision(args.precision)
    targs = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "model"),
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine", weight_decay=args.weight_decay, optim="adamw_torch",
        max_grad_norm=1.0, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=50, report_to=["none"], fp16=fp16, bf16=bf16,
        remove_unused_columns=False, seed=args.seed)

    train_collator = M.Collator(tokenizer, args.max_length, augment=True,
                                word_dropout=args.word_dropout, span_dropout=args.span_dropout,
                                aug_prob=args.aug_prob)
    eval_collator = M.Collator(tokenizer, args.max_length, augment=False)
    callbacks = ([EarlyStoppingCallback(args.early_stopping_patience)]
                 if args.early_stopping_patience > 0 else [])

    trainer = CaptionTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, data_collator=train_collator, eval_collator=eval_collator,
        llrd_factor=args.llrd_factor, compute_metrics=M.compute_metrics, callbacks=callbacks)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print("[train] starting ...")
    trainer.train()

    if args.test_captions:
        print("[test] evaluating with logit voting ...")
        test_records = M.load_records(args.test_captions)
        t_texts, t_labels, t_imgids, t_capidx = M.group_test(test_records, label2id, keys)
        test_ds = Dataset.from_dict({"text": t_texts, "labels": t_labels})
        logits = np.asarray(trainer.predict(test_ds).predictions)
        results = M.vote(logits, t_imgids, t_capidx, t_labels, args.num_captions)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[done] avg_voting_acc={results['avg_voting_acc']:.4f} "
              f"max_voting_acc={results['max_voting_acc']:.4f} -> {args.output_dir}")
    else:
        print(f"[done] model saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
