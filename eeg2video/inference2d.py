# 从EEG数据到视频生成的完整流程

# from tuneavideo.pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline  # 用于从EEG数据生成视频的流水线

import sys
import os
from random import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

from triton.language import dtype

from tuneavideo.models.unet import UNet3DConditionModel  # 用于视频生成的UNet模型
from diffusers import UNet2DConditionModel
from tuneavideo.util import save_videos_grid  # 用于保存视频的函数
import torch  # PyTorch库，用于深度学习模型和张量处理
import numpy as np  # 数组处理库
from einops import rearrange  # 用于张量重排列的库
from sklearn import preprocessing  # 用于数据预处理（标准化）
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP模型

from diffusers import StableDiffusionPipeline, DDIMScheduler,AutoencoderKL  # 用于视频生成的Stable Diffusion模型和调度器
from PIL import Image
import imageio  # 用于生成GIF
from torchvision import transforms
import torch.nn.functional as F

# --------------------------
# 0. 设备设置与路径配置
# --------------------------
device = "cuda"
latents_path = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/test_tensor12n.pt'  # 噪声潜在变量路径
text_emb_path = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/text_embedding_77768_2.npy'  # 文本嵌入路径
video_text = ('../SEED-DV/Video/BLIP-caption/1st_10min.txt')
with open(video_text, 'r') as f:
    text_prompts = [line.strip() for line in f]
# text_prompts = text_prompts[:1]
output_dir = "./40_Classes_woDANA35/"  # 输出目录

# --------------------------
# 1. 加载模型与管道
# --------------------------
# 加载 Stable Diffusion v1-4 预训练模型
model_id = "./checkpoints/stable-diffusion-v1-4"
# 自定义 UNet 配置
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    # 关键修改：适配非对称分辨率
    sample_size=64,  # 原为 (64, 64)
    in_channels=4,
    out_channels=4,
    low_cpu_mem_usage=False,
)

# 自定义 VAE 配置
vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder="vae",
    # 关键修改：关闭默认分辨率检查
    ignore_mismatched_sizes=True,
)
# 初始化管道
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
).to(device)
# 显式转换模型权重
pipe.unet = pipe.unet.half()  # 确保 UNet 权重和偏置为 FP16
pipe.vae = pipe.vae.half()    # 确保 VAE 权重和偏置为 FP16
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,  # 半精度节省显存
#     safety_checker=None  # 禁用安全检查（可选）
# ).to(device)


# # 替换为你的自定义3D管道（根据你的TuneAVideoPipeline实现）
# class TuneAVideoPipeline(StableDiffusionPipeline):
#     # 这里需要你之前定义的TuneAVideoPipeline类代码
#     pass
# # 初始化自定义管道
# pipe = TuneAVideoPipeline(
#     vae=pipe.vae,
#     tokenizer=pipe.tokenizer,
#     unet=pipe.unet,
#     scheduler=DDIMScheduler.from_config(pipe.scheduler.config),
# ).to(device)

# 启用显存优化
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()

# --------------------------
# 2. 加载数据
# --------------------------
# 加载噪声潜在变量 [200, 4, 6, 36, 64]
noisy_latents = torch.load(latents_path).to(device)  # 确保设备一致
print(noisy_latents.mean(),noisy_latents.std())
# noisy_latents = 1/0.18215*noisy_latents
# noisy_latents = noisy_latents.squeeze(0)
print(noisy_latents.shape)
# 插值到64x64
# 计算需要填充的大小
padding_top = (64 - 36) // 2  # 上方填充
padding_bottom = 64 - 36 - padding_top  # 下方填充
noisy_latents = F.pad(noisy_latents, (0, 0, padding_top, padding_bottom), mode='constant', value=0)
noisy_latents = noisy_latents.to(torch.float16)
print("pad后",noisy_latents.shape)

random_latents = torch.randn(10, 6, 4, 64, 64).to(device).to(torch.float16)
print(random_latents.mean(),random_latents.std())

# 加载文本嵌入并调整形状 [200, 77, 768]
text_embeddings = np.load(text_emb_path)
text_embeddings = torch.from_numpy(text_embeddings)  # 转换为PyTorch张量
text_embeddings = text_embeddings.view(-1, 77, 768).to(device)
text_embeddings = text_embeddings.to(torch.float16)
print(text_embeddings.shape)


# --------------------------
# 3. 生成视频帧
# --------------------------
def generate_video_frames(latents, text_emb, idx):
    frames = []
    # 去噪生成
    for i in range(6):
        with torch.no_grad():
            output = pipe(
                latents=latents[i].unsqueeze(0),  # 初始噪声潜在变量 [1,4,6,36,64]
                prompt=text_emb,  # 文本提示
                # height=288,  # 输出图像高度
                # width=512,  # 输出图像宽度
                # num_frames=6,  # 输出帧数
                num_inference_steps=100,  # 去噪步数（可调整）
                guidance_scale=7.5,  # 引导强度（可调整）
                eta=0.0,  # 确定性生成
                output_type="numpy"  # 输出numpy数组
            )
            frame = output.images[0]

        frames.append(frame) # 取第一个batch
    return frames


# --------------------------
# 4. 批量处理并保存GIF
# --------------------------
# for i in range(noisy_latents.shape[0]):
for i in range(10):
    print(f"Processing video {i + 1}/200...")

    # 提取单个样本
    latent = noisy_latents[i]  # [1,6,4,36,64]
    # latent = random_latents[i]  # [1,6,4,36,64]
    # latent = rearrange(latent, 'b f c h w -> (b f) c h w')
    text_emb = text_embeddings[i].unsqueeze(0)  # [1,77,768]

    # 生成帧
    frames = generate_video_frames(latent,text_prompts[i], i)

    # 转换为PIL图像并保存
    gif_path = f"{output_dir}/{i}.gif"
    pil_images = []
    for frame in frames:
        img = Image.fromarray((frame * 255).astype(np.uint8))
        pil_images.append(img)

    # 保存为GIF（帧率可调）
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=500,  # 每帧显示时间（毫秒）
        loop=0  # 无限循环
    )


'''
dana3 - 使用unconditioned latents
dana4 - 1000轮的transformer
dana5 - 改 50推理step 400transformer
dana6 - 100推理 100小lr的transformer
dana7 - d1024的400transformer 
dana8 - 401 d=2048 en3de6  
dana9 - 404 f2
dana10 - 406 f2
dana11 - 406 改成了使用text embedding当做hidden输入

dana12 - 11 改random tensor    incidice修改文件夹-random tensor+text embedding
dana13 - random tensor+eeg embedding
dana14 - transformer latents+random embedding          糊了（12>13>14）说明latent大问题      embedding语义有一点问题但起到作用 
dana15 - vae latents+random embedding                  糊 说明vae latent本身有问题
dana16 - vae noisy latents（tensor1） + random embedding
dana17 - vae noisy latents（tensor1） + eeg embedding
dana18 - vae noisy latents（tensor1） + text embedding
dana19 - vae noisy latents（tensor2） + text embedding     所有都不如随机latents 要解决保存noisylatens
dana20 - 改对的 vae noisy
dana21 - eeg latents(dana9)+text embedding
dana22 - eeg latents(dana11)+text embedding

改了tunemultivideo后的
dana23 - vae noisy latents（tensor3n）+text embedding  (还没换try8模型) 效果还不错 一多半能明确图中目标
dana24 - vae noisy latents（tensor3n）+eeg embedding                  少部分能看出来 说明semantic model有一些效果
dana25 - eeg latents(dana12)+text embedding             trash
dana26 - vae noisy latents（tensor3n）+text embedding(2-sd clip)      和23类似 clip问题不大
dana27 - vae noisy latents（tensor3）+eeg embedding(2-sd clip)   
dana28 - vae noisy latents（tensor4n）+text embedding(2-sd clip)      好于23 26 证明时间步匹配(100)有用（推理用50或200步差异不大）   缩放后效果很好！！
dana29 - vae noisy latents（tensor5n）+text embedding(2-sd clip)      400略好于28的100步 并验证去噪步数也要匹配      缩放后400的并不好
dana30 - vae noisy latents（tensor6n）+text embedding(2-sd clip)      800更不行 加多时间步不对 
dana31 - vae noisy latents（tensor7n）+text embedding(2-sd clip)      50步 和100差不多
dana32 - random tensor+text embedding(2-sd clip)
dana33 - vae noisy latents（tensor8n）+text embedding(2-sd clip)      t=999  糊
dana34 - vae noisy latents（tensor4n）+eeg embedding(2-sd clip)       去噪150 guidance8更专注去噪 eta0.1y引入少量噪声跳出局部最优  对比了各种latent和参数的结果
'''
