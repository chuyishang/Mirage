#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

GPU_ID="$1"

# ——— Hard-coded settings ———
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT="mirage_preliminaries"
DATASET_NAME="blink_jigsaw"                       # to change
RUN_NAME="jigsaw_inference_noise_filler2"           # to change
SETUP="inference_noise_filler2"                     # to change
MAX_TOK=1000
OUTPUT_DIR="./test_outputs"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_NAME}.jsonl"
# ————————————————————————

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Using GPU #$GPU_ID"
echo "Model:             $MODEL"
echo "Run name:          $RUN_NAME"
echo "Setup:             $SETUP"
echo "W&B project:       $PROJECT"
echo "Max new tokens:    $MAX_TOK"
echo "Output file:       $OUTPUT_FILE"
echo

# to change file if needed
python blink_jigsaw_inference.py \
  --model          "$MODEL" \
  --device         cuda \
  --max_new_tokens "$MAX_TOK" \
  --output_file    "$OUTPUT_FILE" \
  --project        "$PROJECT" \
  --run_name       "$RUN_NAME" \
  --dataset_name   "$DATASET_NAME" \
  --setup          "$SETUP"
