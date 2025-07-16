export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="7"

export WANDB_PROJECT="vlm-reason"

    # --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_split.jsonl \
    # --data_path ./data/vsp_spatial_planning/train_split.jsonl \
    # --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_split_same_image.jsonl \

# python src/main.py \
    # --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 15 \
    # --task vsp-spatial-reasoning \
    # --latent_size 4 \
    # --stage stage1 \
    # --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_split_same_image.jsonl \
    # --log_file ./log.txt \
    # --save_model_path ./checkpoints/stage1_split_same_image  \
    # --cache_dir ~/.cache/huggingface/hub \
    # --run_name stage1-split-same-image



# --load_model_path ./checkpoints/stage1_split_same_image \
python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 15 \
    --task vsp-spatial-reasoning \
    --stage stage2 \
    --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_no_image.jsonl \
    --log_file ./log.txt \
    --cache_dir ~/.cache/huggingface/hub \
    --save_model_path /scratch/current/shang/checkpoints/stage2_split_no_image \
    --run_name stage2-split-no-image