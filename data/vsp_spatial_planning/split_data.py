import random
import json
from loguru import logger

random.seed(42)

with open('/home/shang/Mirage/data/vsp_spatial_planning/train_direct.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

logger.info(f"Loaded {len(data)} items")

train_length = int(len(data) * 0.8)

logger.info(f"Splitting into {train_length} train and {len(data) - train_length} val")

random.shuffle(data)

train_data = data[:train_length]
val_data = data[train_length:]

with open('/home/shang/Mirage/data/vsp_spatial_planning/train_split.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('/home/shang/Mirage/data/vsp_spatial_planning/val_split.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')