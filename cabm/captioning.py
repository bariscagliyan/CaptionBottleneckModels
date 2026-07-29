"""Stage 1: multi-caption generation with deterministic taxonomy censoring.

A frozen vision-language model produces K captions per image (one per sampling
temperature). Class names and group/family terms are then removed from each
caption so the downstream text classifier cannot rely on label-name shortcuts.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# --------------------------------------------------------------------------- #
# Taxonomy censoring
# --------------------------------------------------------------------------- #
_PLURAL = r"(?:e?s)?"  # optional plural suffix: "", "s", "es"


def clean_label(label: str) -> str:
    """Normalize a raw class label: drop a leading 'NNN.' index and underscores."""
    return re.sub(r"^\s*\d+\.", "", str(label)).replace("_", " ").strip()


def _is_plural(matched: str) -> bool:
    low = matched.lower()
    return low.endswith("s") and not low.endswith("ss")  # keep 'albatross' singular


def class_name_regex(class_names: List[str]) -> Optional[re.Pattern]:
    """Match class names; comma/semicolon-separated synonyms are matched independently.
     Spaces/hyphens/underscores are interchangeable."""
    sep, pats = r"(?:[\s\-_]+)", []
    for raw in class_names:
        for variant in re.split(r"[;,]", str(raw)):
            toks = [re.escape(t) for t in clean_label(variant).split() if t]
            if toks:
                pats.append(sep.join(toks) + _PLURAL)
    if not pats:
        return None
    pats.sort(key=len, reverse=True)  # longest first: multi-word before sub-words
    return re.compile(r"(?i)\b(?:%s)\b" % "|".join(pats))


def term_regex(terms: Set[str]) -> Optional[re.Pattern]:
    """Match single-word group/family terms with optional plural suffixes."""
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"(?i)\b(?:%s)\b" % "|".join(re.escape(t) + _PLURAL for t in terms))


def load_censor_terms(path: str) -> Set[str]:
    """Load extra terms from JSON: a list, or an object with a 'terms'/'group_terms' key."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("terms") or data.get("group_terms") or []
    return {str(t).strip().lower() for t in data if str(t).strip()}


def censor(text: str, class_re, term_re, replacement: str = "subject") -> str:
    """Replace matched class names / terms with the referent and collapse runs."""
    def sub(m):
        return replacement + ("s" if _is_plural(m.group(0)) else "")

    if class_re is not None:
        text = class_re.sub(sub, text)
    if term_re is not None:
        text = term_re.sub(sub, text)
    tok = re.escape(replacement)
    runs = re.compile(r"(?i)\b%s(s?)\b(?:\s+\b%s(?:s?)\b)+" % (tok, tok))
    return runs.sub(lambda m: replacement + ("s" if m.group(1) else ""), text)


def clean_text(text: str) -> str:
    """Strip markdown artifacts, leading numbering, and collapse whitespace."""
    text = text.replace("**", "").replace("##", "")
    text = re.sub(r"^\s*[1-9]\)\s*", "", text)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_samples(data_root: Optional[str], manifest: Optional[str],
                  image_root: Optional[str]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """Return [(image_path, class_name)] and the sorted list of class names.

    Provide either --data-root (folders 'root/<class>/<image>') or --manifest
    (JSONL of {"image": path, "label": class_name}).
    """
    samples: List[Tuple[Path, str]] = []
    if manifest:
        base = Path(image_root) if image_root else None
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = Path(rec["image"])
                if base and not p.is_absolute():
                    p = base / p
                samples.append((p, str(rec["label"])))
    elif data_root:
        root = Path(data_root)
        for cls in sorted(d.name for d in root.iterdir() if d.is_dir()):
            for fp in sorted((root / cls).iterdir()):
                if fp.suffix.lower() in VALID_EXTS:
                    samples.append((fp, cls))
    else:
        raise ValueError("Provide either --data-root or --manifest.")
    class_names = sorted({c for _, c in samples})
    return samples, class_names


# --------------------------------------------------------------------------- #
# Vision-language model
# --------------------------------------------------------------------------- #
def load_captioner(model_id: str, dtype: str = "bf16", device_map: str = "auto"):
    """Load a Qwen2.5-VL / Qwen3-VL (or generic image-text) model and its processor."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(torch_dtype=torch_dtype, device_map=device_map,
                  trust_remote_code=True, attn_implementation="sdpa")
    mt = getattr(config, "model_type", "")
    model = None
    if mt == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        model = M.from_pretrained(model_id, **kwargs)
    elif mt == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as M
        model = M.from_pretrained(model_id, **kwargs)
    else:
        try:
            from transformers import AutoModelForImageTextToText as M
            model = M.from_pretrained(model_id, **kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
        min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


@torch.inference_mode()
def caption_batch(model, processor, images: List[Image.Image], prompt: str,
                  temperature: float, decoding: dict) -> List[str]:
    """Generate one caption per image at the given temperature."""
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": im},
                                       {"type": "text", "text": prompt}]}]
        for im in images
    ]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
    gen = model.generate(**inputs, do_sample=True, num_beams=1,
                         temperature=temperature, **decoding)
    return processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
