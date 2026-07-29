#!/usr/bin/env bash
# End-to-end CaBM example. Edit DATA_ROOT/TEST_ROOT to point at your dataset,
# then run: bash examples/run_all.sh
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data/train}     # folders: <root>/<class_name>/<image>
TEST_ROOT=${TEST_ROOT:-data/test}
PROMPT=${PROMPT:-prompts/cub_fine_grained.txt}
OUT=${OUT:-runs/demo}

# 1) captions (class names censored by default)
python generate_captions.py --data-root "$DATA_ROOT" --prompt "$PROMPT" \
    --output "$OUT/captions_train.jsonl"
python generate_captions.py --data-root "$TEST_ROOT" --prompt "$PROMPT" \
    --output "$OUT/captions_test.jsonl"

# 2) text classifier
python train_classifier.py \
    --train-captions "$OUT/captions_train.jsonl" \
    --test-captions  "$OUT/captions_test.jsonl" \
    --output-dir "$OUT/classifier"

# 3) concept extraction
python extract_concepts.py \
    --captions "$OUT/captions_train.jsonl" \
    --model-dir "$OUT/classifier" \
    --output-dir "$OUT/concepts"

echo "Done. Concepts in $OUT/concepts/{proposed,clustered}_concepts.json"
