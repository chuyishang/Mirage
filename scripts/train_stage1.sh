export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

export WANDB_PROJECT="vlm-reason"
export RUN_NAME="stage1-test"

python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 16 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage1 \
    --data_path /home/shang/Mirage/data/vsp_spatial_planning/train_direct.jsonl \
    --log_file ./log.txt \
    --save_model_path ./checkpoints/model_stage1_v2  \
    --cache_dir ~/.cache/huggingface/hub
