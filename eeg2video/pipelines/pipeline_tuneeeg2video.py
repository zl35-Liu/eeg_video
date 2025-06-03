# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""
一个用于生成视频的扩散模型管道，基于给定的EEG信号（eeg）生成相应的视频。
它采用了自定义的 EEGencoder 模型进行EEG信号到文本的编码，
并通过 UNet 模型进行图像生成，支持分类器自由引导，允许生成多个视频并返回解码后的结果

在inference里用
"""
import inspect  # 用于检查函数签名
from typing import Callable, List, Optional, Union  # 类型注解
from dataclasses import dataclass  # 用于定义数据类
# from ..models.eeg2text import CLIP as EEGencoder  # 从自定义的EEG编码模型导入CLIP类
# from ..models.train_semantic_predictor import CLIP as EEGencoder  #改成可能是对的引用


import numpy as np  # 用于数值计算
import torch  # 用于深度学习模型和张量操作

# 导入Diffusers和Transformers的工具模块
from diffusers.utils import is_accelerate_available  # 检查是否安装accelerate库
from packaging import version  # 用于版本比较
from transformers import CLIPTextModel, CLIPTokenizer  # 用于CLIP文本模型和标记器

from diffusers.configuration_utils import FrozenDict  # 冻结字典，防止修改
from diffusers.models import AutoencoderKL  # 用于自编码器KL模型
from diffusers.pipeline_utils import DiffusionPipeline  # 用于扩散模型的基类
from diffusers.schedulers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)  # 用于不同类型的调度器（用于控制扩散过程）
from diffusers.utils import deprecate, logging, BaseOutput  # 日志记录和弃用警告
from einops import rearrange  # 用于重新排列张量维度

from EEG2Video.models.unet import UNet3DConditionModel  # 用于3D UNet模型（条件生成）

logger = logging.get_logger(__name__)  # 获取日志记录器


# 定义输出数据类
@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]  # 定义输出的视频数据，可以是torch.Tensor或np.ndarray


# 定义主管道类  实际执行扩散模型视频生成任务的主类
class TuneAVideoPipeline(DiffusionPipeline):            # 父类是 DiffusionPipeline
    _optional_components = []  # 可选组件，暂时为空

    def __init__(
            self,
            vae: AutoencoderKL,  # 自动编码器KL模型
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,  # CLIP标记器
            unet: UNet3DConditionModel,  # UNet3D条件模型
            scheduler: Union[  # 扩散过程的调度器
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
    ):
        super().__init__()  # 调用父类初始化

        # 处理调度器配置的兼容性问题
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        # 检查调度器的clip_sample配置项
        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        # 检查UNet模型的版本兼容性问题
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following models: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly."
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # 注册管道的各个组件
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        # 设置VAE缩放因子（通常为2的幂）
        #通过获取 vae 配置中的 block_out_channels 的长度（可能是某种卷积层的通道数），减去 1 后计算 2 的幂次，得到缩放因子。这个因子可能用于调整生成图像的大小或分辨率
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # 启用VAE切片
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    # 禁用VAE切片
    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    # 启用GPU上的顺序CPU卸载
    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():  # 检查是否可以使用accelerate
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")  # 获取设备

        # 为UNet和VAE模型进行CPU卸载
        for cpu_offloaded_model in [self.unet, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    # 获取执行设备
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # 这个 跟tuneavideo 有改动
    def _encode_eeg(self, model, eeg, device, num_videos_per_eeg, do_classifier_guidance, negative_eeg):
        # 获取EEG数据的嵌入
        eeg_embeddings = model(eeg.to(device)).half()
        #  对齐 CLIP 文本编码规格   77：CLIP 标准上下文长度   768：嵌入维度
        # eeg_embeddings = torch.reshape(eeg_embeddings, [eeg_embeddings.shape[0], 77, 768]).half() # .half() 转为半精度（FP16），节省显存并加速计算

        bs_embed, seq_len, _ = eeg_embeddings.shape
        '''
        沿序列维度（dim=1）重复 num_videos_per_eeg 次
        示例：原始形状 [2,77,768] + num_videos=3 → [2,231,768]'''
        eeg_embeddings = eeg_embeddings.repeat(1, num_videos_per_eeg, 1)  # 将每个 EEG 嵌入重复 num_videos_per_eeg 次，目的是为每个 EEG 信号生成多个视频
        # view 操作将扩展维度转换为批量维度  结果形状：[2*3,77,768] = [6,77,768]
        eeg_embeddings = eeg_embeddings.view(bs_embed * num_videos_per_eeg, seq_len, -1)
        print("eeg emb shape",eeg_embeddings.shape)

        # 如果使用分类器引导，则将无条件嵌入添加到EEG嵌入中.
        if do_classifier_guidance:


            uncond_embeddings = np.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/text_embedding_empty1.npy')
            uncond_embeddings = torch.from_numpy(uncond_embeddings).half().cuda()

            # # 使用标准正态分布随机初始化
            # uncond_embeddings = torch.randn(bs_embed * num_videos_per_eeg, 77, 768, dtype=torch.float16, device=device)
            # # 加入少量的随机噪声（通过strength参数控制）
            # noise = torch.randn_like(uncond_embeddings) * 0.1
            # uncond_embeddings += noise
            # # 标准化：将其转化为均值为0，标准差为1的分布
            # uncond_embeddings = (uncond_embeddings - uncond_embeddings.mean(dim=-1, keepdim=True)) / (
            #             uncond_embeddings.std(dim=-1, keepdim=True) + 1e-5)

            # uncond_embeddings = np.load('/home/v-xuanhaoliu/EEG2Video/Tune-A-Video/negative.npy') # 加载无条件嵌入 作者的是什么
            # uncond_embeddings = torch.zeros(bs_embed * num_videos_per_eeg, 77, 768, dtype=torch.float16, device=device)

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_eeg, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_videos_per_eeg, seq_len, -1)
            print(uncond_embeddings.shape)
            eeg_embeddings = torch.cat([uncond_embeddings, eeg_embeddings])  # 拼接条件/非条件

        return eeg_embeddings

    # 无改动 解码潜在向量为视频
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents  # 缩放潜在向量
        latents = rearrange(latents, "b c f h w -> (b f) c h w")  # 重排维度
        video = self.vae.decode(latents).sample  # 解码
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)  # 重新排列为视频格式
        video = (video / 2 + 0.5).clamp(0, 1)  # 归一化并限制值在0到1之间
        video = video.cpu().float().numpy()  # 转换为numpy数组
        return video

    # 无改动 准备扩散调度器所需的额外参数
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())  # 检查调度器是否接受eta参数
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta  # 设置eta值

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())  # 检查调度器是否接受generator参数
        if accepts_generator:
            extra_step_kwargs["generator"] = generator  # 设置generator
        return extra_step_kwargs

    # 无改动 检查输入的有效性
    def check_inputs(self, eeg, height, width, callback_steps):
        if not isinstance(eeg, torch.Tensor):
            raise ValueError(f"`eeg` has to be of type `torch.Tensor` but is {type(eeg)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # 无改动 根据输入的参数生成或调整  准备扩散模型生成的  ‘初始  潜在向量（randn随机生成）
    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator,
                        latents=None):
        '''
        batch_size：指定了每次生成的批次大小（例如，一次生成多少个视频）
        num_channels_latents：潜变量的通道数（通常与图像的深度、潜在表示的特征维度相关）
        video_length帧数
        height 和 width：视频图像的高和宽（在 VAE 生成过程中，图像分辨率通常会缩小，height // self.vae_scale_factor 代表经过 VAE 处理后图像的分辨率）
        generator：生成潜变量时使用的随机数生成器，可以是一个列表，用于为每个批次生成不同的噪声。
        latents：已存在的潜变量，如果传入，则不重新生成，而是对其进行处理
        '''
        shape = (
        batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有随机生成 生成中不需要
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list): # 如generator是列表，每批次生成一个最后合并
                shape = (1,) + shape[1:] # 将形状调整为只为一个批次生成
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)  # 合并并移动到目标设备
            else:  # 如果 generator 只是单个生成器，则直接生成一个批次的潜在向量
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)

        # 如有 对应inference 对latents处理
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # 缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # 执行扩散过程  调用扩散模型来生成视频
    @torch.no_grad()
    def __call__(  # 调用管道进行生成
            self,
            model, # from tuneavideo.models.eeg_text import CLIP
            #pretrained_eeg_encoder_path = '/home/v-xuanhaoliu/EEG2Video/Tune-A-Video/tuneavideo/models/eeg2text_40_eeg.pt'
            eeg: torch.FloatTensor,   # eeg_test
            video_length: Optional[int],  # inference = 6
            height: Optional[int] = None,  # inference = 288
            width: Optional[int] = None,  # inference = 512
            num_inference_steps: int = 50,  # inference = 100
            guidance_scale: float = 7.5,  # inference = 12.5  指导尺度，用于分类器自由引导
            negative_prompt: Optional[Union[str, List[str]]] = None,  # 负向提示
            num_videos_per_eeg: Optional[int] = 1,  # 每个提示生成的视频数量
            eta: float = 0.0,  # eta（用于DDIM调度器）
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
            latents: Optional[torch.FloatTensor] = None,  # inference导入的‘latents are pre-prepared by Seq2Seq model’
            output_type: Optional[str] = "tensor",  # 输出类型，可以是"tensor"或"numpy"
            return_dict: bool = True,  # 是否返回字典
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,  # 回调函数
            callback_steps: Optional[int] = 1,  # 调用回调的步数间隔
            **kwargs,
    ):
        # 设置默认分辨率
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 检查输入是否有效
        self.check_inputs(eeg, height, width, callback_steps)

        # 定义一些参数
        batch_size = eeg.shape[0]
        device = self._execution_device  # 获取执行设备（GPU或CPU）

        # 判断是否使用分类器自由引导
        do_classifier_free_guidance = guidance_scale > 1.0

        # 用了model得到embedding  model应该是semantic里的 这里跑出了对齐语义的embedding 应当输入模型才对
        eeg_embeddings = self._encode_eeg(model, eeg, device, num_videos_per_eeg, do_classifier_free_guidance,
                                          negative_prompt)
        print("text_emb shape", eeg_embeddings.shape)

        # 设置时间步长
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps    #时间步定义 应等于inference设置100
        print(self.scheduler.timesteps)

        # 准备潜在向量
        num_channels_latents = self.unet.in_channels
        # prepare_latents 实际上是确定一下shape 和 缩放
        latents = self.prepare_latents(     #  搞清1  pre latent的形状
            #形状（R1结果）：(batch*num_videos, 4, video_length, H//8, W//8)
            batch_size * num_videos_per_eeg,
            num_channels_latents,
            video_length,
            height,
            width,
            eeg_embeddings.dtype,   # 用了 model得到的embedding
            device,
            generator,
            latents,    # 用了传参数导入的latents
        )
        latents_dtype = latents.dtype  # 获取潜在向量的数据类型

        # 准备调度器步骤的额外参数
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 执行去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            #  迭代去噪过程
            for i, t in enumerate(timesteps):
                '''
                # 扩展潜在向量，如果使用分类器自由引导
                这样做的目的是为了使用额外的引导信号来修正生成的潜变量。
                分类器自由引导通常是在生成模型时进行的，它通过引入一个无条件噪声（noise_pred_uncond）和一个有条件噪声（noise_pred_text），
                然后通过两者的差异来调整模型的生成输出，从而引导模型更接近目标。
                '''
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # 通过调度器对 latent_model_input 进行缩放（通常会涉及时间步 t），以便输入模型的潜变量更符合生成模型的时间步骤和尺度要求

                '''
                利用 UNet 模型根据输入的潜变量（latent_model_input）和时间步 t，
                以及通过 EEG 编码器生成的嵌入（eeg_embeddings）来预测噪声残差。
                UNet 是用于图像生成的常见神经网络结构，这里它用来预测潜变量在当前时间步的噪声残差。
                '''
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=eeg_embeddings).sample.to(   #eeg embedding的形状
                    dtype=latents_dtype)

                # 如果启用了 CFG，模型会根据无条件噪声和有条件噪声的差异来调整噪声预测
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # 调度器更新
                '''
                通过调度器（scheduler）将噪声预测（noise_pred）应用于当前的潜变量（latents），
                并通过 t 时间步进行更新。调度器控制着去噪的进程，
                它会根据当前的噪声预测更新潜变量，以使其朝着目标图像或视频的方向收敛
                '''
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # 调用回调函数（如果提供）
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # vae decoder
        video = self.decode_latents(latents)

        # 转换为张量
        if output_type == "tensor":
            video = torch.from_numpy(video)

        # 如果不需要返回字典，直接返回视频数据
        if not return_dict:
            return video

        # 返回包含视频的字典对象
        return TuneAVideoPipelineOutput(videos=video)
