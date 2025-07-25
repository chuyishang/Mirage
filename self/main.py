# TODO: add yaml parsing and launch script

import torch
import wandb
import argparse
import random
import os

from loguru import logger
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor, AutoProcessor, Qwen2_5_VLConfig
from qwen_vl_utils import process_vision_info
from trl import SFTConfig, SFTTrainer

# from utils import *
import utils
from trainer import *
from task import task_preprocess_config

utils.seed_everything(42)


def main(args):
    # os.environ['HF_HOME'] = args.cache_dir # ? not sure if this is good

    train_dataset = load_dataset(args.data_path)
    preprocess_function = task_preprocess_config[args.task]
    train_dataset = [preprocess_function(sample) for sample in train_dataset]

    config = Qwen2_5_VLConfig.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        config=config,
        torch_dtype=torch.bfloat16,
        # cache_dir=args.cache_dir,
    )
    if args.stage in ['stage1', 'stage2', 'stage3']:
        # Add latent tokens to tokenizer
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]})
        model.resize_token_embeddings(len(processor.tokenizer))
        latent_pad_id = processor.tokenizer.convert_tokens_to_ids('<|latent_pad|>')
        latent_start_id = processor.tokenizer.convert_tokens_to_ids('<|latent_start|>')
        latent_end_id = processor.tokenizer.convert_tokens_to_ids('<|latent_end|>')
        model.config.latent_token_id = latent_pad_id
        model.config.latent_start_id = latent_start_id
        model.config.latent_end_id = latent_end_id

    image_pad_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    for param in model.visual.parameters():
        param.requires_grad = False


    ### COLLATE FN ###
    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]
        texts = utils.add_latent_tokens(texts, args.latent_size)
        # TODO: add latent pad tokens to texts after or during applying the chat template
        image_inputs, _ = process_vision_info(examples)

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100 # ignore pad tokens
        for id in [vision_start_id, vision_end_id, image_pad_id]:
            labels[labels == id] = -100 # ignore special image tokens
        batch['labels'] = labels

        image_out_mask = utils.create_mask_after_start(batch["input_ids"], latent_start_id, latent_pad_id)
        batch["image_out_mask"] = image_out_mask

        return batch


    ### TRAINING ARGS ###
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        gradient_checkpointing=args.gradient_checkpointing,
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        # learning_rate=2e-4,  # Learning rate for training
        learning_rate=1e-5,
        weight_decay=0.01,
        # lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=20,  # Steps interval for logging
        # eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",
        save_steps=500,
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        # tf32=True,  # Use TensorFloat-32 precision
        # max_grad_norm=0.3,  # Maximum norm for gradient clipping
        # warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name=args.run_name if hasattr(args, 'run_name') else None,
        logging_dir='./logs/',
        logging_strategy='steps',
        # Gradient checkpointing settings
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset


    ### WANDB ###
    wandb.init(
        project="vlm-reason",  # change this
        name=args.run_name,  # change this
        config=training_args,
    )

    ### TRAINER ###
    if args.stage in ['sft']:
        CustomTrainer = SFTTrainer
    elif args.stage in ['stage1']:
        CustomTrainer = CustomTrainerStage1
    elif args.stage in ['stage2']:
        CustomTrainer = CustomTrainerStage2
    # elif args.stage in ['stage3']:
        # CustomTrainer = CustomTrainerStage3
    else:
        raise ValueError(f"Invalid stage: {args.stage}")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--data_path", type=str, default="/home/shang/Mirage/data/vsp_spatial_planning/train_split.jsonl")
    p.add_argument("--output_dir", type=str, default="/scratch/current/shang/checkpoints/sft_vsp_spatial_planning")
    p.add_argument("--task", type=str, default="vsp-spatial-reasoning")
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    # p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", type=bool, default=True)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--cache_dir", type=str, default="~/.cache/huggingface/hub")
    p.add_argument("--run_name", type=str, default="sft_vsp_7-17_lr2e-5")
    p.add_argument("--latent_size", type=int, default=10)
    main(p.parse_args())