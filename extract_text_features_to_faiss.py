from PIL import Image
import requests
import os
import json
from mindspore import context, nn, ops
from mindnlp.transformers import Blip2Processor, Blip2Model, AutoTokenizer, Blip2ForImageTextRetrieval
from tqdm import tqdm
import numpy as np
import faiss
from collections import defaultdict

from mindnlp.transformers.models.blip_2.modeling_blip_2 import Blip2TextModelWithProjection

# 加载模型和处理器
processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
model = Blip2TextModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g")



# 加载COCO数据集的val2014部分
coco_val_images_dir = "/home/ma-user/work/workplace/mindnlp/val2014/val2014"
coco_annotations_file = "/home/ma-user/work/workplace/mindnlp/ana-val2014/annotations/captions_val2014.json"

with open(coco_annotations_file, 'r') as f:
    annotations = json.load(f)

# 重新组织数据结构
image_data = defaultdict(lambda: {"captions": [], "ids": []})
for ann in tqdm(annotations['annotations'], desc="Organizing annotations"):
    image_id = ann['image_id']
    image_data[image_id]["captions"].append(ann['caption'])
    image_data[image_id]["ids"].append(ann['id'])
    if "image_path" not in image_data[image_id]:
        image_data[image_id]["image_path"] = os.path.join(coco_val_images_dir, f"COCO_val2014_{image_id:012d}.jpg")

# 提取所有文本
all_texts = [caption for data in image_data.values() for caption in data["captions"]]

# 特征提取
def extract_text_features(texts, batch_size=128):
    text_features = []
    num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

    for i in tqdm(range(num_batches), desc="Extracting text features", total=num_batches):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        inputs = processor(text=batch_texts, padding=True, return_tensors="ms")
        outputs = model(**inputs)

        batch_features = outputs.text_embeds[:, 0, :]
        batch_features = ops.L2Normalize(axis=-1)(batch_features)
        text_features.append(batch_features.asnumpy())
    
    return np.concatenate(text_features, axis=0)

text_features = extract_text_features(all_texts)

# 存储到Faiss文件中
dimension = text_features.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(text_features)

# 创建元数据
metadata = []
feature_idx = 0
for image_id, data in image_data.items():
    for caption, caption_id in zip(data["captions"], data["ids"]):
        metadata.append({
            "image_id": image_id,
            "id": caption_id,
            "caption": caption,
            "feature_idx": feature_idx
        })
        feature_idx += 1

# 保存Faiss索引和元数据
faiss.write_index(index, "text_features/text_features.index")
with open("text_features/metadata.json", 'w') as f:
    json.dump(metadata, f)

print("Text features and metadata have been saved to Faiss index and JSON file.")
print(f"Total number of images: {len(image_data)}")
print(f"Total number of captions: {len(metadata)}")
