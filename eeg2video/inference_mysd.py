# from tuneavideo.pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline  # 用于从EEG数据生成视频的流水线

import sys
import os
from random import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

import torch  # PyTorch库，用于深度学习模型和张量处理
import numpy as np  # 用于数组处理的库
from einops import rearrange  # 用于张量重排列的库
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel  # 用于生成视频的库

pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
model = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

latents = torch.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/test_tensor9n.pt')
latents = 1/0.18215 * latents
latents = latents.float().to(torch.device('cuda'))  # 将模型加载到GPU上
latents.squeeze_(1)
print(latents.shape)   # 200 4 6 36 64
latents = latents[:1,:, :1, :, :]
latents= rearrange(latents, 'a b c d e -> (a c) b d e')  # 重排列张量维度

print("latent",latents.shape)

textdata = np.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/text_embedding_77768_2.npy')
textdata = rearrange(textdata, 'a d b c  -> (a d) b c')  # 重新排列张量维度
textdata = torch.from_numpy(textdata)  # 转换为PyTorch张量
textdata = textdata.float()  # 确保输入是  float32
textdata = textdata.to(torch.device('cuda'))  # 将模型加载到GPU上
textdata = textdata[:1, :, :]
print("textdata",textdata.shape)

# pipeline_output = image_pipe(torch.randn(1, 4, 64, 64), torch.randn(1, 77, 768))
pipeline_output = image_pipe(latents, textdata)
pipeline_output.images[0]