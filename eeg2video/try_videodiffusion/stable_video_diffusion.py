import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"PyTorch 使用的 CUDA 版本: {torch.version.cuda}")  # 应与 nvcc 版本一致
print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
from diffusers import StableVideoDiffusionPipeline

import imageio
import cv2
from PIL import Image




pipe = StableVideoDiffusionPipeline.from_pretrained(
    "../checkpoints/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()  # 显存优化

# 加载输入图像（调整尺寸为 1024x576）
input_image = Image.open("input.jpg").resize((1024, 576))
prompt = "a city with tall buildings and a street, 4K"
negative_prompt = "low quality, blurry"

# 生成视频（默认生成 25 帧）
video_frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,  # 推理步数（质量 vs 速度）
    num_frames=25,           # 帧数
    fps=10,                  # 帧率
    guidance_scale=12.0,     # 提示词权重（推荐 9-13）
    generator=torch.Generator().manual_seed(42)  # 随机种子
).frames[0]

# 保存为 GIF
imageio.mimsave("output.gif", video_frames, fps=10)

# 保存为 MP4（需安装 OpenCV）

height, width, _ = video_frames[0].shape
video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
for frame in video_frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video.release()