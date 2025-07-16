import json
import argparse
import random

random.seed(42)


def main(args):
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"data length: {len(data)}")

    new_data = []
    if args.type == "same_image":
        # Helper image is the same as input image
        for item in data:
            item["image_output"] = item["image_input"]
            new_data.append(item)
    elif args.type == "shuffle_image":
        # Shuffle only the image_output fields
        image_outputs = [item['image_output'] for item in data]
        random.shuffle(image_outputs)
        new_data = [
            {**item, 'image_output': image_outputs[i]}
            for i, item in enumerate(data)
        ]
    elif args.type == "noise_image":
        for i,item in enumerate(data):
            item["image_output"] = f"/home/shang/Mirage/data/noise_images/noise_{i}.png"
            new_data.append(item)
    elif args.type == "gray_image":
        for i,item in enumerate(data):
            item["image_output"] = f"/home/shang/Mirage/data/vsp_spatial_planning/img/gray.png"
            new_data.append(item)
    elif args.type == "no_image":
        for i,item in enumerate(data):
            del item["image_output"]
            new_data.append(item)
    else:
        raise ValueError(f"Invalid type: {args.type}")

    with open(f"./data/vsp_spatial_planning/{args.output_name}.jsonl", "w") as f:
        for item in new_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="/home/shang/Mirage/data/vsp_spatial_planning/train_split.jsonl")
    p.add_argument("--output_name", type=str)
    # p.add_argument("--output_path", type=str, default="/home/shang/Mirage/data/vsp_spatial_planning/train_split_same_image.jsonl")
    p.add_argument("--type", type=str)
    main(p.parse_args())
