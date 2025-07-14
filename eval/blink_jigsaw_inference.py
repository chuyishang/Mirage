import json
import re
from pathlib import Path
from PIL import Image
import torch
import wandb
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import numpy as np

def extract_boxed_answer(text: str):
    m = re.search(r"\\boxed\{([AB])\}", text)
    if m:
        return m.group(1)
    m2 = re.search(r"(?m)^[ \t]*([AB])[ \t]*(?:$|\n)", text)
    if m2:
        return m2.group(1)
    return None

def evaluate_jigsaw(model_name, device, max_new_tokens, output_file):
    SUBTASK = "Jigsaw"
    dataset = load_dataset("BLINK-Benchmark/BLINK", SUBTASK)
    val_data = dataset["val"]
    total = len(val_data)
    correct_count = 0

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map=None
    ).to(device)
    model.eval()

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")

    pbar = tqdm(total=total, desc="Running inference")
    for idx, sample in enumerate(val_data):
        ref_img = sample["image_1"].convert("RGB")
        patch_A = sample["image_2"].convert("RGB")
        patch_B = sample["image_3"].convert("RGB")

        raw_ans = sample.get("answer", "")
        correct = raw_ans.strip().strip("()")  # 'A' or 'B'
        patch = patch_A if correct == "A" else patch_B
        helper_img = ref_img.copy()
        x0 = ref_img.width  - patch.width
        y0 = ref_img.height - patch.height        
        helper_img.paste(patch, (x0, y0))
        
        #grey_patch = Image.new("RGB", (ref_img.width, ref_img.height), (128, 128, 128))

        noise_arr = np.random.randint(
                    low=0, high=256,
                    size=(ref_img.height, ref_img.width, 3),
                    dtype=np.uint8
        )
        noise_img = Image.fromarray(noise_arr)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text",  "text": sample["prompt"]},
                {"type": "image", "image": ref_img},
                {"type": "image", "image": patch_A},
                {"type": "image", "image": patch_B},
                #{"type": "text",  "text": "For a hint, here is the first image with missing part filled up:"},
                {"type": "text",  "text": "Here is some thinking space:"},
                {"type": "image", "image": noise_img},
                # {"type": "text",  "text": "Here is the first image again:"},
                # {"type": "image", "image": ref_img},
                {"type": "text",  "text": "Please provide your final answer by putting the choice in \\boxed{} with A or B only."}
            ],
        }]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            return_tensors="pt",
            padding=True
        ).to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        raw = processor.decode(out_ids[0], skip_special_tokens=True)

        assistant_token = "assistant\n"
        if assistant_token in raw:
            reply = raw.split(assistant_token, 1)[1]
        else:
            reply = raw

        out_f.write(json.dumps({
            "index": idx,
            "prediction_raw": reply
        }, ensure_ascii=False) + "\n")

        pred = extract_boxed_answer(reply)
        if pred == correct:
            correct_count += 1
        pbar.update(1)
    pbar.close()

    out_f.close()
    acc = correct_count / total if total else 0.0
    print(f"Accuracy: {correct_count}/{total} = {acc*100:.2f}%")
    wandb.log({"accuracy": acc})

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str, required=True,
                   help="Hugging Face model ID, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device",         type=str, default="cuda",
                   help="Torch device, e.g. 'cuda' or 'cpu'")
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="Max tokens to generate")
    p.add_argument("--output_file",    type=str, default="assistant_outputs.jsonl",
                   help="Where to save raw outputs")
    p.add_argument("--project",        type=str, required=True,
                   help="W&B project name")
    p.add_argument("--run_name",       type=str, required=True,
                   help="W&B run name")
    p.add_argument("--dataset_name",   type=str, required=True,
                   help="Logical dataset name for logging")
    p.add_argument("--setup",          type=str, required=True,
                   help="Setup description for logging")
    args = p.parse_args()

    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "model": args.model,
            "dataset": args.dataset_name,
            "setup": args.setup,
        }
    )

    evaluate_jigsaw(
        model_name     = args.model,
        device         = args.device,
        max_new_tokens = args.max_new_tokens,
        output_file    = args.output_file
    )