export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$1"
# export RUN_NAME="$2"
# export DATASET="$3"
# export EPOCHS="${4:-10}"
# export LATENT_SIZE="${5:-4}"
export RUN_NAME="self_test"
export DATASET="train_split"
export EPOCHS=10
export LATENT_SIZE=4

export WANDB_PROJECT="vlm-reason"

echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using RUN_NAME: $RUN_NAME"
echo "Using DATASET: $DATASET"
echo "Using EPOCHS: $EPOCHS"
echo "Using LATENT_SIZE: $LATENT_SIZE"

    # --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_split.jsonl \
    # --data_path ./data/vsp_spatial_planning/train_split.jsonl \
    # --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_split_same_image.jsonl \
    # --log_file ./log.txt \

python -m self.main \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs $EPOCHS \
    --task vsp-spatial-reasoning \
    --latent_size $LATENT_SIZE \
    --stage stage1 \
    --data_path /home/shang/Mirage/data/vsp_spatial_planning/$DATASET.jsonl \
    --save_model_path /scratch/current/shang/checkpoints/stage1_$RUN_NAME  \
    --cache_dir ~/.cache/huggingface/hub \
    --run_name $RUN_NAME-stage1
