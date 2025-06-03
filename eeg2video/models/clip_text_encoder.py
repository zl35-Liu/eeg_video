"""
自己写的clip文本编码器 用于生成semantic里用的text_embedding
"""

from transformers import CLIPProcessor, CLIPModel,CLIPTokenizer,CLIPTextModel
import torch
import numpy as np

def compute_global_mean_var(data):
    flattened_data = data.flatten()  # 展平为1D数组
    global_mean = np.mean(flattened_data)
    global_var = np.var(flattened_data)
    return global_mean, global_var




pretrained_model_path = "../checkpoints/stable-diffusion-v1-4"  # 预训练模型路径- SD
model_path = "./clip-vit-large-patch14"
# 加载预训练的 CLIP 模型和处理器
# model = CLIPModel.from_pretrained(model_path)
# model.eval()

tokenizer = CLIPTokenizer.from_pretrained(model_path)  # 文本分词器
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")  # 文本编码器 测试自己找的clip是否有区别
# text_encoder = CLIPTextModel.from_pretrained(model_path)


# 应输出非全零值（类似[-0.023, 0.017, ...]）

# 读取文本文件，每行是一个句子
text_file_path = "/home/bcilab/Ljx/EEG2Video-main/SEED-DV/Video/BLIP-caption/all.txt"  # 包含1-7
# text_file_path = "E:/TJ/store/EEG2Video-main/SEED-DV/Video/BLIP-caption/all.txt"  # 包含1-7
# text_file_path = "/home/bcilab/Ljx/EEG2Video-main/SEED-DV/Video/BLIP-caption/1st_10min.txt"  # 包含1-7

# 从文本文件读取所有句子
with open(text_file_path, 'r') as f:
    texts = [line.strip() for line in f.readlines()]
text_empty = ''    # 用于生成空文本的emb


# 标准流程：自动执行L2归一化
# texts = ["your sentences here"]
# inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
# with torch.no_grad():
#     text_features = model.get_text_features(**inputs)  # 自动执行L2归一化
#
# # 验证统计信息
# print(text_features.shape)
# print(text_features[0][0])
# print("范围:", text_features[0].min().item(), text_features[0].max().item())  # 应接近[-1, 1]
# print("范数:", text_features[0].norm(dim=-1).mean().item())  # 应接近1.0
# emb = text_features.cpu().detach().numpy()
# text_mean, text_var = compute_global_mean_var(emb)
# print("文本embedding总体均值:", text_mean)
# print("文本embedding总体方差:", text_var)

input_ids0 = tokenizer(texts, max_length=77, padding="max_length",
        truncation=True,
        return_tensors="pt").input_ids
# attention_mask0 = input.attention_mask
input_empty = tokenizer(text_empty, max_length=77, padding="max_length",
        truncation=True,
        return_tensors="pt")
input_ids = input_empty.input_ids
attention_mask = input_empty.attention_mask

# 使用模型的 text_encoder 获取文本嵌入
with torch.no_grad():
    text_output = text_encoder(input_ids0)
    text_output1 = text_encoder(input_ids,
                               attention_mask=attention_mask)

#text_embeddings 是一个包含文本嵌入的张量
# print(text_output.shape)

out = text_output[0]
out = out.cpu().detach().numpy()
print(out.shape)
# out = out.view(40, 5, 77, 768)
# print(out.shape)


out1 = text_output1.last_hidden_state
print(out1)
print(out1.shape)
# 将文本嵌入保存为 NumPy 数组


text_mean, text_var = compute_global_mean_var(out)
print("文本embedding总体均值:", text_mean)
print("文本embedding总体方差:", text_var)
out = torch.from_numpy(out)
# 取第一个样本的第一个token 验证\
# 若仍存在数值异常，手动标准化
# normalized_embeddings = (
#     out - out.mean(dim=-1, keepdim=True)
# ) / (out.std(dim=-1, keepdim=True) + 1e-8)
sample_embedding = out[0].unsqueeze(0)
print(f"单token均值: {sample_embedding.mean().item():.4f}")
print(f"单token方差: {sample_embedding.var().item():.4f}")
# print(f"单token范数: {sample_embedding.norm(dim=-1).meaan().item():.4f}")
print(f"文本输入范围: [{out.min():.2f}, {out.max():.2f}]")  # 应≈[-3,3]
print(f"文本嵌入范数: {out.norm(dim=-1).mean():.2f}")  # 应≈1.0

out = out.cpu().detach().numpy()
np.save("text_embedding_masked2.npy", out)


"""
1  clip vl14  1.txt
2  sd clip  1.txt
3  masked-加了mask的1.txt  empty1-加了mask的空字符
4  masked1-all.txt
5  masked2 -all  使用diffusion相同的encoder 并检验输出分布和模型hidden一致
"""
