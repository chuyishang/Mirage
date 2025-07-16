#!/usr/bin/env bash
set -euo pipefail

# if [ $# -ne 1 ]; then
  # echo "Usage: $0 <GPU_ID>"
  # exit 1
# fi

# GPU_ID="$1"
# MODEL_NAME="$2"
# RUN_NAME="$3"

GPU_ID="4,5"
MODEL_NAME="sft_vsp_spatial_planning/checkpoint-1000"
RUN_NAME="sft_vsp_spatial_planning_ckpt1000"

# ——— Hard-coded settings ———
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL="/home/shang/Mirage/checkpoints/$RUN_NAME/"
MODEL="/scratch/current/shang/checkpoints/$MODEL_NAME"
# MODEL="/scratch/current/shang/checkpoints/stage2_train_same_7-16"
PROJECT="vlm-reason"
DATASET_NAME="/home/shang/Mirage/data/vsp_spatial_planning/val_split.jsonl"                       # to change
DATA="/home/shang/Mirage/data/vsp_spatial_planning/val_split.jsonl"
RUN_NAME="$RUN_NAME"
SETUP="$RUN_NAME"
MAX_TOK=1000
OUTPUT_DIR="/home/shang/Mirage/eval/outputs/$RUN_NAME"
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
