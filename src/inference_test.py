import json
import re
import tarfile
from pathlib import Path
from PIL import Image
import torch
import wandb
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import numpy as np

def load_dataset(path: str):
    path = Path(path)
    if path.suffix in {".gz", ".tgz"} and tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name.endswith(".json"):
                    return json.load(tf.extractfile(member))
        raise RuntimeError("No .json found in tar.gz")
    else:
        # assume plain JSONL
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]

def extract_boxed_answer(text: str):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1) if m else None

def normalize_moves(seq: str):
    mapping = {
        'UP':'U','U':'U',
        'DOWN':'D','D':'D',
        'LEFT':'L','L':'L',
        'RIGHT':'R','R':'R'
    }
    tokens = re.split(r'[\s,;]+', (seq or "").strip().upper())
    return [mapping[t] for t in tokens if t in mapping]

def evaluate(json_path, model_name, device, max_new_tokens, output_file):
    data = load_dataset(json_path)
    total = len(data)
    correct = 0

    root_dir = Path(json_path).parent

    processor = AutoProcessor.from_pretrained(model_name)
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map=None
    ).to(device)
    model.eval()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_path, "w", encoding="utf-8")

    for idx, ex in enumerate(tqdm(data, desc="Running inference")):
        prompt_text = ex["text_input"].strip()

        def resolve_img_path(s):
            s = s.replace("\\", "")
            i = s.find("img/")
            if i < 0:
                raise ValueError(f"Expected 'img/' in {s}")
            return root_dir / s[i:]

        img1 = Image.open(resolve_img_path(ex["image_input"])).convert("RGB")
        # noise_arr = np.random.randint(
        #     low=0, high=256,
        #     size=(img1.height, img1.width, 3),
        #     dtype=np.uint8
        # )
        # noise_img = Image.fromarray(noise_arr)
        img2 = Image.open(resolve_img_path(ex["image_output"])).convert("RGB")

        # ground truth
        gt_raw   = extract_boxed_answer(ex["text_output"])
        gt_moves = normalize_moves(gt_raw)

        # 3) build the same 'messages' list
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text.split("Here is the map: ")[0]
                                + "Here is the map: "
                    },
                    {"type": "image", "image": img1},
                    {"type": "text",  "text": "\nHere is my reasoning path: "},
                    # {"type": "text",  "text": "\nHere is some reasoning space: "},
                    {"type": "image", "image": img2},
                    {
                        "type": "text",
                        "text": "Please provide your action plan. "
                                "The final answer MUST BE put in \\boxed{}."
                    },
                ],
            }
        ]

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
            max_new_tokens = max_new_tokens,
            do_sample      = False,
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

        pred_raw   = extract_boxed_answer(reply)
        pred_moves = normalize_moves(pred_raw)
        if pred_moves == gt_moves:
            correct += 1

    out_f.close()
    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc*100:.4f}%")
    wandb.log({"accuracy": acc})
    wandb.finish()
    return acc

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data",           type=str,  required=True,
                   help="Path to train_direct.jsonl or .tar.gz")
    p.add_argument("--model",          type=str,
                   default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device",         type=str,  default="cuda")
    p.add_argument("--max_new_tokens", type=int,  default=32)
    p.add_argument("--output_file",    type=str,  
                   default="assistant_outputs.jsonl",
                   help="Where to save the assistant’s raw outputs")
    p.add_argument("--project",        type=str,  required=True,
                   help="W&B project name")
    p.add_argument("--run_name",       type=str,  required=True,
                   help="W&B run name")
    p.add_argument("--dataset_name",   type=str,  required=True,
                   help="Logical name of the dataset (for logging)")
    p.add_argument("--setup",          type=str,  required=True,
                   help="Setup description (for logging)")
    args = p.parse_args()

     # W&B init
    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "model": args.model,
            "dataset": args.dataset_name,
            "setup": args.setup,
        }
    )

    evaluate(
        json_path      = args.data,
        model_name     = args.model,
        device         = args.device,
        max_new_tokens = args.max_new_tokens,
        output_file    = args.output_file
    )
