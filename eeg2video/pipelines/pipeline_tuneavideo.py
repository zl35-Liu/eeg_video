# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""
用于生成视频的扩散模型管道，基于给定的文本提示（prompt）生成相应的视频。
它采用了 CLIP 模型进行文本编码，并通过 UNet 模型进行图像生成，
支持分类器自由引导，允许生成多个视频并返回解码后的结果
"""
import inspect  # 用于检查函数签名
from typing import Callable, List, Optional, Union  # 类型注解
from dataclasses import dataclass  # 用于定义数据类

import numpy as np  # 用于数值计算
import torch  # 用于深度学习模型和张量操作

from diffusers.utils import is_accelerate_available  # 检查是否安装accelerate库
from packaging import version  # 用于版本比较
from transformers import CLIPTextModel, CLIPTokenizer  # 用于CLIP文本模型和标记器

from diffusers.configuration_utils import FrozenDict  # 冻结字典，防止修改
from diffusers.models import AutoencoderKL  # 用于自编码器KL模型
from diffusers.pipeline_utils import DiffusionPipeline  # 用于扩散模型的基类
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)  # 用于不同类型的调度器（用于控制扩散过程）
from diffusers.utils import deprecate, logging, BaseOutput  # 日志记录和弃用警告
from einops import rearrange  # 用于重新排列张量维度

from ..models.unet import UNet3DConditionModel  # 用于3D UNet模型（条件生成）

logger = logging.get_logger(__name__)  # 获取日志记录器

# 定义输出数据类
@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]  # 定义输出的视频数据，可以是torch.Tensor或np.ndarray


# 定义主管道类
class TuneAVideoPipeline(DiffusionPipeline):
    _optional_components = []  # 可选组件，暂时为空

    def __init__(
        self,
        vae: AutoencoderKL,  # 自动编码器KL模型
        text_encoder: CLIPTextModel,  # CLIP文本编码器
        tokenizer: CLIPTokenizer,  # CLIP标记器
        unet: UNet3DConditionModel,  # UNet3D条件模型
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],  # 扩散过程的调度器
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

        # 为UNet、文本编码器和VAE模型进行CPU卸载
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
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

    # 编码输入文本
    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # 编码文本
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        # 检查是否有文本被截断
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        # 获取attention mask（如果可用）
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        # 获取文本嵌入
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # 为每个prompt生成多个视频时，重复文本嵌入
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # 获取无条件嵌入（用于分类器自由引导）
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:" 
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # 为每个生成的样本重复无条件嵌入
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # 保存无条件嵌入（可以用于调试）
            uncond_numpy = uncond_embeddings.detach().cpu().numpy()
            np.save('negative.npy', uncond_numpy)

            # 合并无条件和条件嵌入，以便进行一次推理（而不是两次）
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # 解码潜在向量为视频
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents  # 缩放潜在向量
        latents = rearrange(latents, "b c f h w -> (b f) c h w")  # 重排维度
        video = self.vae.decode(latents).sample  # 解码
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)  # 重新排列为视频格式
        video = (video / 2 + 0.5).clamp(0, 1)  # 归一化并限制值在0到1之间
        video = video.cpu().float().numpy()  # 转换为numpy数组
        return video

    # 准备扩散调度器所需的额外参数
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())  # 检查调度器是否接受eta参数
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta  # 设置eta值

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())  # 检查调度器是否接受generator参数
        if accepts_generator:
            extra_step_kwargs["generator"] = generator  # 设置generator
        return extra_step_kwargs

    # 检查输入的有效性
    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # 准备潜在向量
    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # 缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # 执行扩散过程
    @torch.no_grad()
    def __call__(  # 调用管道进行生成
        self,
        prompt: Union[str, List[str]],  # 输入文本（提示）
        video_length: Optional[int],  # 视频长度
        height: Optional[int] = None,  # 输出图像高度
        width: Optional[int] = None,  # 输出图像宽度
        num_inference_steps: int = 50,  # 推理步骤数
        guidance_scale: float = 7.5,  # 指导尺度，用于分类器自由引导
        negative_prompt: Optional[Union[str, List[str]]] = None,  # 负向提示
        num_videos_per_prompt: Optional[int] = 1,  # 每个提示生成的视频数量
        eta: float = 0.0,  # eta（用于DDIM调度器）
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 随机数生成器
        latents: Optional[torch.FloatTensor] = None,  # 初始潜在向量
        output_type: Optional[str] = "tensor",  # 输出类型，可以是"tensor"或"numpy"
        return_dict: bool = True,  # 是否返回字典
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,  # 回调函数
        callback_steps: Optional[int] = 1,  # 调用回调的步数间隔
        **kwargs,
    ):
        # 设置默认高度和宽度
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 检查输入是否有效
        self.check_inputs(prompt, height, width, callback_steps)

        # 定义一些参数
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device  # 获取执行设备（GPU或CPU）

        # 判断是否使用分类器自由引导
        do_classifier_free_guidance = guidance_scale > 1.0

        # 编码输入的文本
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 设置时间步长
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 准备潜在向量
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype  # 获取潜在向量的数据类型

        # 准备调度器步骤的额外参数
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 执行去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 扩展潜在向量，如果使用分类器自由引导
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 预测噪声残差
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

                # 进行引导
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # 计算之前的噪声样本 x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # 调用回调函数（如果提供）
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 后处理（解码潜在向量为视频）
        video = self.decode_latents(latents)

        # 转换为张量
        if output_type == "tensor":
            video = torch.from_numpy(video)

        # 如果不需要返回字典，直接返回视频数据
        if not return_dict:
            return video

        # 返回包含视频的字典对象
        return TuneAVideoPipelineOutput(videos=video)
