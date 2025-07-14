#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

GPU_ID="$1"

# ——— Hard-coded settings ———
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL="/home/shang/Mirage/checkpoints/stage2_split"
PROJECT="vlm-reason"
DATASET_NAME="/home/shang/Mirage/data/vsp_spatial_planning/val_split.jsonl"                       # to change
DATA="/home/shang/Mirage/data/vsp_spatial_planning/val_split.jsonl"
RUN_NAME="stage2_200_base"           # to change
SETUP="stage2_200_base"                     # to change
MAX_TOK=1000
OUTPUT_DIR="/home/shang/Mirage/eval/outputs/stage2_200_base"
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
python ./eval/vsp_inference.py \
  --model          "$MODEL" \
  --device         cuda \
  --max_new_tokens "$MAX_TOK" \
  --output_file    "$OUTPUT_FILE" \
  --project        "$PROJECT" \
  --run_name       "$RUN_NAME" \
  --dataset_name   "$DATASET_NAME" \
  --data           "$DATA" \
  --setup          "$SETUP"
