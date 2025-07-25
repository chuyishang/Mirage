import json
import wandb
import argparse
import torch

import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import bitsandbytes as bnb

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_linear_schedule_with_warmup
from qwen_vl_utils import process_vision_info
from loguru import logger

def extract_label(sample):
    raw_ans = sample.get("answer", "")
    return raw_ans.strip().strip("()").upper()

class JigsawDataset(Dataset):
    def __init__(self, split, processor, K_latent, helper_emb_file, device):
        self.data = load_dataset("BLINK-Benchmark/BLINK", "Jigsaw")[split]
        emb = torch.load(helper_emb_file)
        self.helper_embs = emb.to(device)
        self.processor = processor
        self.K = K_latent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        ref = sample["image_1"].convert("RGB")
        A = sample["image_2"].convert("RGB")
        B = sample["image_3"].convert("RGB")
        label = extract_label(sample)
        helper_emb = self.helper_embs[idx]
        return {
            "prompt": sample["prompt"],
            "ref": ref,
            "A": A,
            "B": B,
            "helper_emb": helper_emb,
            "label": label
        }

def collate_fn(batch, processor, K, latent_id, device):
    inputs_list = []
    for sample in batch:
        msg = [
            {"type": "text",  "text": sample["prompt"]},
            {"type": "image", "image": sample["ref"]},
            {"type": "image", "image": sample["A"]},
            {"type": "image", "image": sample["B"]},
            {"type": "text",  "text": f"For a hint, imagine the missing part filled up:"},
            {"type": "text",  "text": " ".join(["<latent_pad>"] * K)},
            {"type": "text",  "text": "Please answer with A or B."}
        ]
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True
        )
        vis_inputs, _ = process_vision_info([{"role":"user","content":msg}])
        proc = processor(text=[prompt], images=vis_inputs, return_tensors="pt", padding=True)
        batch_t = {k: v.to(device) for k, v in proc.items()}
        tokenized_label = processor.tokenizer([sample['label']], add_special_tokens=False).input_ids[0]
        full_labels = torch.full_like(batch_t["input_ids"], -100)
        L = len(tokenized_label)
        full_labels[:, -L:] = torch.tensor(tokenized_label, device=device).unsqueeze(0)
        batch_t['labels'] = full_labels
        batch_t['helper_emb'] = sample['helper_emb'].half().to(device)
        inputs_list.append(batch_t)

    batched = {}
    keys = inputs_list[0].keys()
    for k in keys:
        if k == 'helper_emb':
            batched[k] = torch.stack([d[k] for d in inputs_list], dim=0)
        else:
            batched[k] = torch.cat([d[k] for d in inputs_list], dim=0)
    return batched


def train(args):
    # Initialize WandB
    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            'model': args.model,
            'batch_size': args.batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
            'epochs_stage1': args.epochs_stage1,
            'epochs_stage2': args.epochs_stage2,
            # 'epochs_stage3': args.epochs_stage3,
            'K': args.K,
            'lambda_latent': args.lambda_latent,
            'max_grad_norm': args.max_grad_norm,
            'dataset': args.dataset,
            'setup': args.setup,
        }
    )

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<latent_pad>"]})
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=None
    )
    for p in model.visual.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable()
    model.to(args.device)
    model.resize_token_embeddings(len(processor.tokenizer))
    latent_id = processor.tokenizer.convert_tokens_to_ids('<latent_pad>')
    # IDs for vision‐related tokens (used to carve out “text” vs “image”)
    # vs_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    # ve_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    # ip_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')

    dataset = JigsawDataset(
        split='val',
        processor=processor,
        K_latent=args.K,
        helper_emb_file=args.helper_emb_file,
        device=args.device
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, args.K, latent_id, args.device)
    )

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    model.train()
    #  Stage 1
    total_steps = len(loader) * args.epochs_stage1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    for epoch in tqdm(range(args.epochs_stage1),
                      desc="Stage1 Epochs",
                      leave=True):
        batch_pbar = tqdm(loader,
                          desc=f" Stage1 E{epoch+1}/{args.epochs_stage1}",
                          leave=False)
        for step, batch in enumerate(batch_pbar, 1):

            attn_mask = batch['attention_mask']

            out = model(
                input_ids=batch['input_ids'],
                attention_mask=attn_mask,
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True
            )
            # Stage 1:
                # Mean pool last layer -> Loss wrt image helper embedding
                # 
                # 0.1 * CE + latent
            # Input: input <latent><latent><latent> Output: answer
            # Loss over "answer"
            logits_final = out.logits[0, -1]
            target_id   = batch['labels'][0, -1]
            ce_loss     = F.cross_entropy(
                logits_final.unsqueeze(0), 
                target_id.unsqueeze(0)
            )

            last_hid = out.hidden_states[-1] 
            mask = (batch['input_ids'] == latent_id)
            eps = 1e-8
            pred = (
                (last_hid * mask.unsqueeze(-1))      
                .sum(dim=1)                          
                / (mask.sum(dim=1, keepdim=True) + eps)      
            )            

            helper_emb = batch['helper_emb']
            lat_loss = (1 - F.cosine_similarity(pred.float(), helper_emb.float(), dim=-1)).mean()  # tensor latent alignment loss
            total_loss = ce_loss + args.lambda_latent * lat_loss
            # backward & accumulate
            (total_loss/args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # log per step
            wandb.log({'stage1_ce': ce_loss.item(), 'stage1_latent': lat_loss.item(), 'stage1_total': total_loss.item()})
            batch_pbar.set_postfix(
                ce=f"{ce_loss.item():.3f}",
                latent=f"{lat_loss.item():.3f}",
                total=f"{total_loss.item():.3f}"
            )
    # save stage1
    stage1_path = Path(args.output_dir)/'stage1'; stage1_path.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(stage1_path)
    processor.save_pretrained(stage1_path)

    # Stage 2
    scheduler2 = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(loader)*args.epochs_stage2)
    for epoch in tqdm(range(args.epochs_stage2),
                      desc="Stage2 Epochs",
                      leave=True):
        batch_pbar = tqdm(loader,
                          desc=f" Stage2 E{epoch+1}/{args.epochs_stage2}",
                          leave=False)
        for step, batch in enumerate(batch_pbar, 1):
            
            attn_mask = batch['attention_mask']
            
            # uncomment if we want answers to not see image
            # ids        = batch['input_ids'] 
            # image_pos  = (ids == vs_id) | (ids == ve_id) | (ids == ip_id)
            # attn_mask = batch['attention_mask'].clone() 
            # attn_mask = attn_mask.masked_fill(image_pos, 0) 
            
            out = model(
                input_ids=batch['input_ids'],
                attention_mask=attn_mask,
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True
            )
            logits_final = out.logits[0, -1]
            target_id   = batch['labels'][0, -1]
            loss     = F.cross_entropy(
                logits_final.unsqueeze(0), 
                target_id.unsqueeze(0)
            )
            (loss/args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler2.step()
                optimizer.zero_grad()
            wandb.log({'stage2_ce': loss.item()})
            batch_pbar.set_postfix(ce=f"{loss.item():.3f}")
    # save stage2
    stage2_path = Path(args.output_dir)/'stage2'; stage2_path.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(stage2_path)
    processor.save_pretrained(stage2_path)

    # # Stage 3
    # scheduler3 = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(loader)*args.epochs_stage3)
    # for epoch in tqdm(range(args.epochs_stage3),
    #                   desc="Stage2 Epochs",
    #                   leave=True):
    #     batch_pbar = tqdm(loader,
    #                       desc=f" Stage2 E{epoch+1}/{args.epochs_stage3}",
    #                       leave=False)
    #     for step, batch in enumerate(batch_pbar, 1):
            
    #         attn_mask = batch['attention_mask']
    #         # uncomment if we want answers to not see image
    #         # ids        = batch['input_ids'] 
    #         # image_pos  = (ids == vs_id) | (ids == ve_id) | (ids == ip_id)
    #         # attn_mask = batch['attention_mask'].clone() 
    #         # attn_mask = attn_mask.masked_fill(image_pos, 0) 
            
    #         out = model(
    #             input_ids=batch['input_ids'],
    #             attention_mask=attn_mask,
    #             pixel_values=batch['pixel_values'],
    #             image_grid_thw=batch.get('image_grid_thw'),
    #             output_hidden_states=True,
    #             return_dict=True
    #         )
    #         logits_final = out.logits[0, -1]
    #         target_id   = batch['labels'][0, -1]
    #         loss     = F.cross_entropy(
    #             logits_final.unsqueeze(0), 
    #             target_id.unsqueeze(0)
    #         )
    #         (loss/args.gradient_accumulation_steps).backward()
    #         if step % args.gradient_accumulation_steps == 0:
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             scheduler3.step()
    #             optimizer.zero_grad()
    #         wandb.log({'stage3_ce': loss.item()})
    #         batch_pbar.set_postfix(ce=f"{loss.item():.3f}")
    # # save stage3
    # stage3_path = Path(args.output_dir)/'stage3'; stage3_path.mkdir(exist_ok=True, parents=True)
    # model.save_pretrained(stage3_path)
    # processor.save_pretrained(stage3_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--epochs_stage1', type=int, default=3)
    parser.add_argument('--epochs_stage2', type=int, default=3)
    # parser.add_argument('--epochs_stage3', type=int, default=3)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--lambda_latent', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient clipping norm')
    parser.add_argument('--helper_emb_file', type=str, required=True, help='Path to precomputed helper embeddings .pt')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--setup', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--run_name', required=True)
    args = parser.parse_args()

    train(args)