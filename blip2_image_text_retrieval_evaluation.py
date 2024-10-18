import os
import json
import faiss
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
from mindspore import ops
from mindnlp.transformers import Blip2Processor, Blip2ForImageTextRetrieval, Blip2VisionModelWithProjection
import mindspore
import mindnlp.core.nn
print("开始加载模型和处理器")


itm_processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
itm_model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
vision_model = Blip2VisionModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g")
print("模型和处理器加载完成")

print("开始加载文本特征和元数据")
index = faiss.read_index("text_features/text_features.index")
with open("text_features/metadata.json", 'r') as f:
    metadata = json.load(f)
print(f"文本特征和元数据加载完成，元数据长度：{len(metadata)}")

# 创建image_id到metadata索引的映射
image_id_to_indices = {}
for idx, item in enumerate(metadata):
    image_id = item['image_id']
    if image_id not in image_id_to_indices:
        image_id_to_indices[image_id] = []
    image_id_to_indices[image_id].append(idx)


def extract_image_features(image_path):
    print(f"正在提取图像特征：{image_path}")
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="ms")
    outputs = vision_model(**inputs)
    image_embeds = outputs.image_embeds
    print(f"图像特征形状：{image_embeds.shape}")
    return image_embeds


def compute_similarity(image_embeds, text_features):
    print("开始计算相似度")
    text_features = mindspore.tensor(text_features)
    similarity = ops.matmul(image_embeds.squeeze(0), text_features.T)
    max_similarity = similarity.max(axis=0)
    print(f"相似度计算完成，最大相似度形状：{max_similarity.shape}")
    print(f"相似度统计：最小值 {max_similarity.min().item():.4f}, 最大值 {max_similarity.max().item():.4f}, 平均值 {max_similarity.mean().item():.4f}")
    return max_similarity


def retrieve_top_k(image_embeds, k=128):
    print(f"开始检索前{k}个最相似的文本")
    max_similarity = compute_similarity(
        image_embeds, index.reconstruct_n(0, index.ntotal))
    top_k_indices = max_similarity.argsort()[-k:][::-1]
    top_k_scores = max_similarity[top_k_indices]
    print(f"检索完成，前5个索引及其分数：")
    for idx, score in zip(top_k_indices[:5], top_k_scores[:5]):
        print(f"索引 {idx}: 分数 {score.item():.4f}")
    return top_k_indices


def compute_itm_scores(image, texts, batch_size=48):
    print("开始计算ITM和ITC组合分数")
    all_scores = []
    num_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总批次数

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = itm_processor(
            images=[image] * len(batch_texts),
            text=batch_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="ms"
        )

        # 计算ITM分数
        itm_out = itm_model(**inputs, use_image_text_matching_head=True)
        itm_scores = ops.softmax(itm_out.logits_per_image, axis=1)[:, 1]

        # 计算ITC分数
        itc_out = itm_model(**inputs, use_image_text_matching_head=False)
        itc_scores = itc_out.logits_per_image.squeeze()

        # 由于ITC分数是二维的，我们需要选择对角线元素 mindspore 不支持 pytorch中的 diag 所以这样实现
        k = mindspore.Tensor(0, mindspore.int32)
        padding_value = mindspore.Tensor(0, itc_scores.dtype)
        itc_scores_diag = ops.matrix_diag_part(itc_scores, k, padding_value)

        # 组合ITM和ITC分数
        combined_scores = itm_scores + itc_scores_diag

        all_scores.append(combined_scores)

        if i == 0:
            for j, score in enumerate(combined_scores[:5]):
                print(f"  文本 {j+1}: {score.asnumpy().item():.4f}")

        current_batch = i // batch_size + 1
        print(
            f"已处理 {current_batch}/{num_batches} 批次 ({min(i+batch_size, len(texts))}/{len(texts)} 个文本)")

    combined_scores = ops.concat(all_scores, axis=0)
    print(f"组合分数计算完成，分数形状：{combined_scores.shape}")
    print(
        f"组合分数统计：最小值 {combined_scores.min().asnumpy().item():.4f}, "
        f"最大值 {combined_scores.max().asnumpy().item():.4f}, "
        f"平均值 {combined_scores.mean().asnumpy().item():.4f}")
    return combined_scores

def evaluate_retrieval(image_paths, k=128):
    results = []
    for idx, image_path in enumerate(tqdm(image_paths, desc="评估图像")):
        print(f"\n开始评估第{idx+1}张图像：{image_path}")
        image = Image.open(image_path).convert('RGB')
        image_embeds = extract_image_features(image_path)
        top_k_indices = retrieve_top_k(image_embeds, k)

        top_k_texts = [metadata[int(idx)]['caption'] for idx in top_k_indices]
        itm_scores = compute_itm_scores(image, top_k_texts)

        reranked_indices = itm_scores.argsort()[::-1]
        reranked_metadata = [
            metadata[int(top_k_indices[idx])] for idx in reranked_indices]

        ground_truth_id = int(os.path.basename(
            image_path).split('_')[-1].split('.')[0])
        retrieved_image_ids = [item['image_id'] for item in reranked_metadata]

        result = {
            'image_id': ground_truth_id,
            'retrieved': retrieved_image_ids
        }
        results.append(result)

        r_at_1 = int(retrieved_image_ids[0] == ground_truth_id)
        r_at_5 = int(ground_truth_id in retrieved_image_ids[:5])
        r_at_10 = int(ground_truth_id in retrieved_image_ids[:10])

        print(
            f"图像 {idx+1}/{len(image_paths)}: R@1: {r_at_1}, R@5: {r_at_5}, R@10: {r_at_10}")
        print(f"地面真实ID: {ground_truth_id}")
        print("检索到的前5个图像ID及其ITM分数:")
        for i, (ret_id, score) in enumerate(zip(retrieved_image_ids[:5], itm_scores[reranked_indices[:5]])):
            print(f"  Rank {i+1}: 图像ID {ret_id}, ITM分数 {score.item():.4f}")

    return results


def compute_metrics(results):
    print("开始计算总体指标")
    r_at_1 = sum(result['retrieved'][0] == result['image_id']
                 for result in results) / len(results)
    r_at_5 = sum(result['image_id'] in result['retrieved'][:5]
                 for result in results) / len(results)
    r_at_10 = sum(result['image_id'] in result['retrieved'][:10]
                  for result in results) / len(results)

    print(f"指标计算完成：R@1: {r_at_1:.4f}, R@5: {r_at_5:.4f}, R@10: {r_at_10:.4f}")
    return {
        'R@1': r_at_1,
        'R@5': r_at_5,
        'R@10': r_at_10
    }


def main():
    print("开始主评估流程")
    coco_val_images_dir = "/home/ma-user/work/workplace/mindnlp/val2014/val2014"
    image_paths = [os.path.join(coco_val_images_dir, f) for f in os.listdir(
        coco_val_images_dir) if f.endswith('.jpg')]
    print(f"找到 {len(image_paths)} 张图像")

    results = evaluate_retrieval(image_paths[:200])
    print(f"评估完成，结果数量：{len(results)}")

    metrics = compute_metrics(results)

    print("最终评估结果:")
    print(f"R@1: {metrics['R@1']:.4f}")
    print(f"R@5: {metrics['R@5']:.4f}")
    print(f"R@10: {metrics['R@10']:.4f}")


if __name__ == "__main__":
    main()
