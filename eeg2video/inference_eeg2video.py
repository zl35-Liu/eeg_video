#从EEG数据到视频生成的完整流程

# from tuneavideo.pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline  # 用于从EEG数据生成视频的流水线

import sys
import os
from random import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

from triton.language import dtype

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline  # 改成对的路径
from tuneavideo.models.unet import UNet3DConditionModel  # 用于视频生成的UNet模型
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from tuneavideo.util import save_videos_grid,ddim_inversion  # 用于保存视频的函数
import torch  # PyTorch库，用于深度学习模型和张量处理
# from tuneavideo.models.eeg_text import CLIP  # 用于EEG到文本的映射的模型
from EEG2Video.models.train_semantic_predictor import CLIP,CLIP1  #改成可能是对的引用  用于EEG到文本的映射的模型
import numpy as np  # 数组处理库
from einops import rearrange  # 用于张量重排列的库
from sklearn import preprocessing  # 用于数据预处理（标准化）
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP模型


def schedule_alpha_bar(t, T, schedule_type="linear"):
    """计算累积噪声保留系数α_bar_t（与扩散模型训练时一致）"""
    if schedule_type == "linear":
        beta_start = 0.0001
        beta_end = 0.02
        beta_t = beta_start + (t / T) * (beta_end - beta_start)
        alpha_bar_t = torch.prod(1 - torch.linspace(beta_start, beta_end, t))
    elif schedule_type == "cosine":
        # 使用Improved DDPM的余弦调度
        s = 0.008  # 防止β_t过小的偏移量
        alpha_bar_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
    else:
        raise ValueError("Unsupported schedule type")
    return alpha_bar_t


def add_noise_to_video_latent(z_0, t, T=1000, schedule_type="linear", time_smooth=True):
    """
    对视频潜在表示进行加噪
    参数:
        z_0 (torch.Tensor): 原始视频潜在，形状 [B, C, T, H, W] (e.g. B=1, C=4, T=16, H=64, W=64)
        t (int): 加噪步数（范围1~T）
        T (int): 总扩散步数
        schedule_type (str): 噪声调度类型（"linear"或"cosine"）
        time_smooth (bool): 是否对噪声的时间维度进行平滑（减少视频闪烁）
    """
    B, C, num_frames, H, W = z_0.shape

    # 1. 计算噪声调度系数
    alpha_bar_t = schedule_alpha_bar(t, T, schedule_type)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

    # 2. 生成高斯噪声（考虑时间连续性）
    if time_smooth:
        # 方法1: 生成低频噪声（时间维度上平滑）
        # 生成基础噪声 [B, C, 1, H, W] 并沿时间复制
        base_noise = torch.randn(B, C, 1, H, W, device=z_0.device)
        epsilon = base_noise.repeat(1, 1, num_frames, 1, 1)
        # 添加时间抖动噪声
        temporal_noise = 0.1 * torch.randn(B, C, num_frames, H, W, device=z_0.device)
        epsilon += temporal_noise
    else:
        # 方法2: 完全独立噪声（可能导致帧间闪烁）
        epsilon = torch.randn_like(z_0)

    # 3. 应用重参数化公式加噪
    z_t = sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * epsilon

    return z_t

# 预训练EEG编码器模型路径  来自semantic
# pretrained_eeg_encoder_path = '/home/v-xuanhaoliu/EEG2Video/Tune-A-Video/tuneavideo/models/eeg2text_40_eeg.pt'
pretrained_eeg_encoder_path = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/semantic_predictor_f3.pt'


# 加载预训练的EEG编码器模型（CLIP）
model = CLIP1(77*768)  # 初始化CLIP模型
# model = CLIP()  # 初始化CLIP模/型
# model.load_state_dict(torch.load(pretrained_eeg_encoder_path, map_location=lambda storage, loc: storage)['state_dict'])  # 加载模型权重
model.to(torch.device('cuda'))  # 将模型加载到GPU上
model.eval()  # 将模型设置为评估模式，关闭dropout等训练时特有的行为

# EEG数据路径和维度
eeg_data_path = "/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/DE_1per1s/sub1.npy.npy"# your own data path for eeg data  应该是DE 给model用的
text_path = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/text_embedding_masked.npy'
EEG_dim = 62*5                             # the dimension of an EEG segment
eegdata = np.load(eeg_data_path)      # 加载EEG数据

textdata = np.load(text_path)
# textdata = textdata[:200]
# textdata = rearrange(textdata, '(a d) b c  -> a d (b c)',a=40,d=5)  # 重新排列张量维度

# random_eeg = torch.randn(200,310).float()
# random_eeg = random_eeg.to(torch.device('cuda'))

# 目标标签（GT_label），表示每个EEG样本的类别
GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])

# chosed_label = [1, 10, 12, 16, 19, 23, 25, 31, 34, 39]
chosed_label = [i for i in range(1, 41)]  # 定义所选标签（chosed_label），用于从EEG数据中提取对应的部分

# 按照选定的标签，提取EEG数据的子集
EEG = []
for i in range(6):   # 前6组   前6组作为训练集
    indices = [list(GT_label[i]).index(element) for element in chosed_label]  # 找到所选标签在GT_label中的索引位置
    chosed_eeg = eegdata[i][indices,:]  # 从EEG数据中提取对应的标签数据
    EEG.append(chosed_eeg)  # 将提取的EEG数据添加到列表
EEG = np.stack(EEG, axis=0)  # 将所有EEG数据按第一个维度拼接成一个数组

# 提取测试数据，按选定标签进行处理
# test_indices = [list(GT_label[6]).index(element) for element in chosed_label]
test_indices = [list(GT_label[0]).index(element) for element in chosed_label]
'''
test_indices0 = [24, 31, 37, 1, 29, 5, 23, 18, 34, 32, 11, 12, 3, 13, 19, 0, 25, 2, 9, 21, 
                33, 20, 10, 16, 17, 6, 8, 28, 7, 30, 26, 35, 4, 36, 22, 14, 38, 40, 39, 15] 
                
                121-125 5:14  156-160 6:46   186-190 
'''
print(len(test_indices))
# eeg_test = eegdata[6][test_indices, :]  # 提取第7组（索引为6）数据   第7组作为test
print("incidice时shape ",eegdata.shape)
eeg_test = eegdata[0][test_indices, :]  # 暂时没做第七组latent 先用训的第一组生成

# textdata = textdata[test_indices, :]
textdata = rearrange(textdata, 'a d b c  -> (a d) b c')  # 重新排列张量维度
textdata = torch.from_numpy(textdata)  # 转换为PyTorch张量
textdata = textdata.float()  # 确保输入是  float32
textdata = textdata.to(torch.device('cuda'))  # 将模型加载到GPU上

eeg_test = torch.from_numpy(eeg_test)  # 转换为PyTorch张量
print("eeg_test转换前 ",eeg_test.shape)
eeg_test = rearrange(eeg_test, 'a b c d e -> (a b) c (d e)')  # 重排列张量的维度
eeg_test = torch.mean(eeg_test, dim=1).resize(eeg_test.shape[0], EEG_dim)  # 计算每个通道的均值，调整张量尺寸

# 转换EEG数据为PyTorch张量并重排列
EEG = torch.from_numpy(EEG)   # 转换为PyTorch张量
print(EEG.shape)
# id = 1
# for i in range(40):
#     EEG[:,i,...] = id
#     id += 1
EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')    # 重排列维度
print(EEG.shape)
EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], EEG_dim)  # 计算均值，调整尺寸

# 标准化EEG数据
scaler = preprocessing.StandardScaler().fit(EEG)  # 初始化标准化器并拟合EEG数据
EEG = scaler.transform(EEG)  # 标准化EEG数据
EEG = torch.from_numpy(EEG).float().cuda()  # 转换为浮动类型的PyTorch张量，并将其移至GPU
eeg_test = scaler.transform(eeg_test)  # 标准化测试数据
eeg_test = torch.from_numpy(eeg_test).float().cuda()  # 转换为PyTorch张量，并移至GPU
print("eeg_test转换后 ",eeg_test.shape)
print("text_data转换后 ",textdata.shape)

# 加载UNet3D模型
pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"  # 预训练模型路径- SD
my_model_path = "/home/bcilab/Ljx/EEG2Video-main/EEG2Video/outputs/18"  # 训练好的模型路径 最后是epoch5
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet',use_memory_efficient_attention=True,gradient_checkpointing=True,torch_dtype=torch.float16).to('cuda')  # 加载UNet模型
# unet = UNet2DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')  # 2d
# unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder='unet').to('cuda')  # 尝试 元模型
# unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')  # 加载UNet模型

generator = torch.Generator(device=torch.device('cuda')).manual_seed(33)
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path,generator=generator, unet=unet, torch_dtype=torch.float16).to("cuda")  # 创建视频生成流水线
# text_encoder = pipe.components["text_encoder"]
print("Pipeline components keys:", pipe.components.keys())
print(hasattr(pipe, 'text_encoder'))  # 应输出True
pipe.enable_xformers_memory_efficient_attention()  # 启用xformers内存高效注意力
pipe.enable_vae_slicing()  # 启用VAE切片优化
ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
ddim_inv_scheduler.set_timesteps(50)

# this are latents with DANA, these latents are pre-prepared by Seq2Seq model
# latents_add_noise = np.load('./tuneavideo/models/latent_add_noise.npy')  # 加载带噪声的潜在变量
# latents_add_noise = torch.from_numpy(latents_add_noise).half()  # 转换为半精度浮点数的PyTorch张量
# latents_add_noise = rearrange(latents_add_noise, 'a b c d e -> a c b d e')  # 重排列张量维度
'''
# this are latents w/o DANA, these latents are pre-prepared by Seq2Seq model
#  sub1  1.mp4 的200段2s
latents = np.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/MyTransformer_pytorch-main/latents_no_DANA12.npy')  # 加载不带噪声的潜在变量
latents = torch.from_numpy(latents).half()  # 转换为半精度浮点数的PyTorch张量
# latents = rearrange(latents, 'a b c d e -> a c b d e')  # 重排列张量维度
latents= rearrange(latents, '(a k) b c d e -> a k b c d e',a=40,k=5)  # 重排列张量维度
latents= latents[test_indices,:]
latents= rearrange(latents, 'a k b c d e -> (a k) b c d e',a=40,k=5)  # 重排列张量维度
print("latents shape",latents.shape)

random_tensor = torch.randn(200, 4, 6, 36, 64).half()  # 生成随机张量
'''


vaes = torch.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/vae_latents.pt')  # 15 测试视频latent
vaes = vaes.squeeze(1)
print("VAE latent范围:", vaes.min(), vaes.max())  # 预期范围 [-1,1] 或 [0,1]
vaes = 1/0.185 * vaes
print("vae shape",vaes.shape)
print("vae均0 ",vaes.mean()," vae方差1  ",vaes.std())  # 检查是否归一
# print("vae before",vaes)
# vaes = rearrange(vaes, 'a b c d e f -> a (b c) d e f')  # 重新排列张量维度
# vaes = rearrange(vaes, '(a k) b c d e f -> a k (b c) d e f',a=40,k=5)  # 重新排列张量维度
# # vaes = vaes[test_indices,:]
# vaes = rearrange(vaes, 'a k b d e f -> (a k) b d e f',a=40,k=5)  # 重新排列张量维度
# vaes = vaes[:,:,:1,:,:]  # 单帧用
# print("vae incidice",vaes)
vaes = vaes.half()  # 确保输入是 float32
# print(latents.shape)  # 输出潜在变量的形状  [200, 4, 6, 36, 64]
print(eeg_test.shape)  # 输出测试数据的形状  [200, 310]


noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")  # 噪声调度
noise = torch.randn_like(vaes).to(vaes.device)
bsz = vaes.shape[0]
timesteps = torch.randint(0, 50, (bsz,), device=vaes.device).long()
print(timesteps)

 # 前向扩散过程：添加噪声到潜在向量
noisy_latents = noise_scheduler.add_noise(vaes, noise, timesteps)
print("noisy shape",noisy_latents.shape)


# ddims=[]
# for i in range(10):
#     ddim = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=vaes[i],
#                                 num_inv_steps=100, prompt="")[-1 ].to(vaes.dtype)
#     ddims.append(ddim.squeeze(0))
# ddim_inv_latent = torch.stack(ddims)
# # ddim_inv_latent = 1/0.18215*ddim_inv_latent
# print("ddim均0 ",ddim_inv_latent.mean()," ddim方差1  ",ddim_inv_latent.std())  # 检查是否归一
#
#
# # ddim_inv_latent = torch.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/outputs/18/inv_latents/ddim_latent-500.pt')
# # ddim_inv_latent = ddim_inv_latent.repeat(200,1,1,1,                                                                                                           1)
# print("ddim shape ",ddim_inv_latent.shape)

inv_latent = torch.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/ddim.pt')
inv_latent = inv_latent.squeeze(1)
print("inv shape",inv_latent.shape)

# Ablation, inference w/o Seq2Seq and w/o DANA
woSeq2Seq = False  # 设置是否去除Seq2Seq部分
woDANA = True  # 设置是否去除DANA部分


# 生成视频并保存
for i in range(0,200):  # 循环200次，每次生成一个视频
    if woSeq2Seq:  # 如果禁用Seq2Seq
        video = pipe(model, eeg_test[i:i+1,...], latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = '40_Classes_woSeq2Seq'  # 保存的文件名
    elif woDANA:  # 如果禁用DANA
        # eeg_test是脑电200段 latents是
        video = pipe(model,textdata[i:i+1,...], latents=inv_latent[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=50, guidance_scale=12.5).videos
        savename = '40_Classes_woDANA37'  # 保存的文件名
    else:  # 如果没有禁用Seq2Seq和DANA
        # video = pipe(model, eeg_test[i:i+1,...], latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = '40_Classes_Fullmodel'  # 保存的文件名
    save_videos_grid(video, f"./{savename}/{i}.gif")  # 保存生成的视频为GIF格式


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


使用微调后的sd
dana36 - vae noisy latents（tensor4n）+text embedding(2-sd clip)  
dana37 - ddim latent +text embedding(2-sd clip)       
'''
