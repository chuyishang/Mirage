from PIL import Image
import datasets


# class JigsawDataset(datasets.Dataset):
    # def __init__(self, split, processor, K_latent, helper_emb_file, device):
        # self.data = load_dataset("BLINK-Benchmark/BLINK", "Jigsaw")[split]
        # emb = torch.load(helper_emb_file)
        # self.helper_embs = emb.to(device)
        # self.processor = processor
        # self.K = K_latent

    # def __len__(self):
        # return len(self.data)

    # def __getitem__(self, idx):
        # sample = self.data[idx]
        # ref = sample["image_1"].convert("RGB")
        # A = sample["image_2"].convert("RGB")
        # B = sample["image_3"].convert("RGB")
        # label = extract_label(sample)
        # helper_emb = self.helper_embs[idx]
        # return {
            # "prompt": sample["prompt"],
            # "ref": ref,
            # "A": A,
            # "B": B,
            # "helper_emb": helper_emb,
            # "label": label
        # }

def single_input_image_preprocess_function(sample):
    # Load images
    
    image = Image.open(sample["image_input"]).convert("RGB") 
    helper_emb = Image.open(sample["helper_emb"]).convert("RGB")

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": helper_emb},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations


def single_input_image_preprocess_function(sample):
    # Load images
    
    image = Image.open(sample["image_input"]).convert("RGB") 
    image_output = Image.open(sample["image_output"]).convert("RGB")

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations

def multiple_input_images_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    image_output = Image.open(sample["image_output"]).convert("RGB")

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "image", "image": image_output}, 
                {"type": "text", "text": sample["text_output"]}
                ],
        },
    ]

    return conversations

task_preprocess_config = {
    'vsp-spatial-reasoning': single_input_image_preprocess_function,
    'vsp-spatial-planning': single_input_image_preprocess_function,
    'blink-jigsaw': multiple_input_images_preprocess_function,
    'sat': multiple_input_images_preprocess_function,
}