import torch
import random
import numpy as np
import json
from datasets import Dataset

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def add_latent_tokens(text, k):
    # """Adds k fixed length latent tokens to the text"""
    # sep_token = "<|im_start|>assistant"
    # assert sep_token in text
    # t1, t2 = text.split(sep_token)
    # latent_tokens = ["<|latent_pad|>"] * k
    # latent_tokens_str = " ".join(latent_tokens)
    # return t1 + sep_token + latent_tokens_str + t2

def add_latent_tokens(texts: list[str], k: int, mode: str = "answer_prefix") -> list[str]:
    """Adds k fixed length latent tokens to the text"""
    new_texts = []
    for text in texts:
        if mode == "question_suffix":
            sep_token = "<|im_end|>\n<|im_start|>assistant\n"
        elif mode == "answer_prefix":
            sep_token = "<|im_start|>assistant"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        assert sep_token in text
        t1, t2 = text.split(sep_token)
        latent_tokens = ["<|latent_pad|>"] * k
        latent_tokens_str = " ".join(latent_tokens)
        new_texts.append(t1 + sep_token + latent_tokens_str + t2)
    return new_texts


def create_mask_after_start(
    input_ids: torch.Tensor,
    start_token: int,
    to_replace_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> after the first <image_start_token> has appeared,
    and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_start_token: the token ID that marks the start of an image chunk
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    batch_size, seq_len = input_ids.shape # (B, S)
    mask = torch.zeros_like(input_ids) # (B, S)

    for i in range(batch_size):
        seq = input_ids[i]
        # Find first occurrence of image_start_token
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == start_token:
                first_start_pos = j
                break
        
        if first_start_pos == -1:
            continue
        
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == to_replace_token:
                mask[i, k] = 1

    return mask
    

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)