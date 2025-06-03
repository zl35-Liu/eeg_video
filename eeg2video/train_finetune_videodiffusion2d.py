# 导入必要的库和模块

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

import math  # 数学运算
import os  # 操作系统相关功能
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
print(torch.cuda.is_available())
import torch.nn.functional as F  # PyTorch神经网络函数
import torch.utils.checkpoint  # 梯度检查点

import diffusers  # 扩散模型库
import transformers  # 自然语言处理模型库
from accelerate import Accelerator  # 分布式训练加速库
from accelerate.logging import get_logger  # 获取日志记录器
from accelerate.utils import set_seed  # 设置随机种子
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel  # 扩散模型组件
from diffusers.optimization import get_scheduler  # 学习率调度器
from tqdm import tqdm  # 进度条显示
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP模型

from tuneavideo.models.unet import UNet3DConditionModel  # 自定义3D UNet模型
from tuneavideo.data.dataset import TuneAVideoDataset, TuneMultiVideoDataset  # 自定义数据集类

from tuneavideo.util import save_videos_grid, ddim_inversion  # 保存视频网格和DDIM反转
from einops import rearrange  # 张量操作
from PIL import Image


# 设备配置
device = "cuda"
dtype = torch.float16  # 半精度加速

# 路径配置
pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
batch_size = 1  # 根据显存调整
num_train_timesteps = 1000  # 扩散模型总步数

# 加载 VAE 编码器
vae = AutoencoderKL.from_pretrained(
    pretrained_model_path,
    subfolder="vae",
    torch_dtype=dtype
).to(device)
# 加载文本编码器
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_path,
    subfolder="text_encoder",
    torch_dtype=dtype
).to(device)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_path,
    subfolder="tokenizer"
)
# 加载噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained(
    pretrained_model_path,
    subfolder="scheduler"
)

video_text = ('../SEED-DV/Video/BLIP-caption/1st_10min.txt')
with open(video_text, 'r') as f:
    text_prompts = [line.strip() for line in f]
# 初始化数据集
dataset = TuneMultiVideoDataset(
    video_path ='../SEED-DV/Video/1.mp4',
    prompt=text_prompts,
    width=512,
    height=288,
    sample_frame_rate=8,
    n_sample_frames=6,
)
# 对文本提示进行分词编码
dataset.prompt_ids = tokenizer(
    list(dataset.prompt),
    max_length=tokenizer.model_max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).input_ids
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
)

# 生成 Noisy Latents 的核心逻辑
def generate_noisy_latents(i,batch):
    """
    输入:
        batch: 来自 TuneMultiVideoDataset 的批次字典，包含
            - pixel_values: [B, T, H, W, C] (T=6帧)
            - input_ids: [B, max_text_len]
    输出:
        noisy_latents: [B, T, 4, 36, 64]
        text_embeddings: [B, max_text_len, 768]
    """
    # --------------------------
    # 1. 编码文本
    # --------------------------
    # text_inputs = tokenizer(
    #     batch["text"],
    #     padding="max_length",
    #     max_length=77,
    #     return_tensors="pt"
    # ).to(device)
    # text_inputs = batch["prompt_ids"]

    # with torch.no_grad():
    #     text_embeddings = text_encoder(text_inputs)[0]  # [B, 77, 768]

    # --------------------------
    # 2. 编码视频帧到潜在空间
    # --------------------------
    videos = batch[i]["pixel_values"].to(device, dtype)  # [B, T, H, W, C]
    B, T = videos.shape[:2]

    # 将视频帧展平为 [B*T, H, W, C]
    frames = videos.reshape(-1, *videos.shape[2:])  # [B*T, H, W, C]
    frames = frames.permute(0, 1, 3, 2)  # [B*T, C, H, W]
    print(frames.shape)
    assert frames.shape[1] == 3, "通道数必须为3 (RGB)"

    # VAE 编码
    with torch.no_grad():
        latents = vae.encode(frames).latent_dist.sample()  # [B*T, 4, 36, 64]
    latents = latents * vae.config.scaling_factor  # 缩放
    print("缩放因子",vae.config.scaling_factor)

    # 恢复形状为 [B, T, 4, 36, 64]
    clean_latents = latents.reshape(B, T, 4, 36, 64)

    # --------------------------
    # 3. 添加扩散过程噪声
    # --------------------------
    # 随机采样时间步
    # timesteps = torch.randint(
    #     0,
    #     noise_scheduler.config.num_train_timesteps,
    #     (B,),
    #     device=device
    # ).long()
    # timesteps = torch.linspace(0,999,100).long().to(latents.device) # 另一种写法
    timesteps = torch.full((B,), 500, device=latents.device, dtype=torch.long)  # 固定时间步为100 测试
    print(timesteps)

    # 生成噪声
    noise = torch.randn_like(clean_latents)  # [B, T, 4, 36, 64]

    # 添加噪声 (Forward Diffusion Process)
    noisy_latents = noise_scheduler.add_noise(
        clean_latents,
        noise,
        timesteps
    )
    return noisy_latents
    # return {
    #     "noisy_latents": noisy_latents.cpu().float(),
    #     "text_embeddings": text_embeddings.cpu().float(),
    #     "timesteps": timesteps.cpu()
    # }


output_dir = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents'
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(output_dir, exist_ok=True)
latents = []
print(dataloader.__len__())
for batch_idx, batch in enumerate(tqdm(dataloader)):
    print(batch_idx)
    if batch_idx ==10:
        break
    # print(batch.keys())
    # print(batch["pixel_values"].shape)
    results = generate_noisy_latents(batch_idx,batch)
    latents.append(results )



all_latents = torch.stack(latents, dim=0)
all_latents = all_latents.squeeze(1)
print(all_latents.shape)
torch.save(all_latents, '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/test_tensor14n.pt')
"""
10n - 保存随机步骤的noisy_latents 1个
11n - 保存100固定步骤的noisy_latents 10个
12n - 保存999固定步骤的noisy_latents 10个
13n - 保存10固定步骤的noisy_latents 10个
14n - 保存500固定步骤的noisy_latents 10个
"""
