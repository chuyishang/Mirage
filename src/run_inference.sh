#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

GPU_ID="$1"

# ——— Hard-coded settings ———
DATA="../data/vsp_spatial_planning/val_split.jsonl" # to change sometimes
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT="mirage_preliminaries"
DATASET_NAME="vsp_spatial_planning_200" # to change sometimes
RUN_NAME="vsp_inference_with_helper" # to change
SETUP="inference_with_helper" # to change
MAX_TOK=1000
OUTPUT_DIR="../test_outputs"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_NAME}.jsonl"
# ————————————————————————

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Using GPU #$GPU_ID"
echo "Dataset path:      $DATA"
echo "Dataset name:      $DATASET_NAME"
echo "Model:             $MODEL"
echo "Run name:          $RUN_NAME"
echo "Setup:             $SETUP"
echo "W&B project:       $PROJECT"
echo "Max new tokens:    $MAX_TOK"
echo "Output file:       $OUTPUT_FILE"
echo

python inference_test.py \
  --data           "$DATA" \
  --model          "$MODEL" \
  --device         cuda \
  --max_new_tokens "$MAX_TOK" \
  --output_file    "$OUTPUT_FILE" \
  --project        "$PROJECT" \
  --run_name       "$RUN_NAME" \
  --dataset_name   "$DATASET_NAME" \
  --setup          "$SETUP"