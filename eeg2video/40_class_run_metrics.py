import numpy as np
from transformers import pipeline  # 用于加载预训练的Transformer模型
from typing import Callable, List, Optional, Union  # 用于类型注解，增强代码可读性
from transformers import ViTImageProcessor, ViTForImageClassification  # 用于图像分类的ViT（Vision Transformer）模型
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification  # 用于视频分类的VideoMAE模型
from transformers import AutoProcessor, CLIPVisionModelWithProjection  # CLIP模型，用于计算图像的特征向量
from PIL import Image  # 用于图像处理
import torch  # 深度学习框架，进行模型推理和张量操作
from einops import rearrange  # 用于张量维度重排列
from torchmetrics.functional import accuracy  # 用于计算准确率等常见评估指标
import torch.nn.functional as F  # 包含许多常用的操作，如损失函数等
from transformers import logging  # 用于设置transformers库的日志等级
from skimage.metrics import structural_similarity as ssim  # 用于计算SSIM（结构相似性指标）
import imageio  # 用于读取和保存图像/视频
from PIL import Image  # 用于图像处理

logging.set_verbosity_error()  # 设置日志等级，只显示错误信息，避免输出过多无关信息


# 定义clip_score类，用于计算图像相似度，基于CLIP模型
class clip_score:
    # 预定义的一些类标签（与图像相关）
    predefined_classes = [
        'an image of people',
        'an image of a bird',
        'an image of a mammal',
        'an image of an aquatic animal',
        'an image of a reptile',
        'an image of buildings',
        'an image of a vehicle',
        'an image of a food',
        'an image of a plant',
        'an image of a natural landscape',
        'an image of a cityscape',
    ]

    def __init__(self,
                 device: Optional[str] = 'cuda',  # 默认使用GPU，如果不可用会自动转为CPU
                 cache_dir: str = '.cache'  # 指定缓存目录
                 ):
        self.device = device  # 设置设备，通常为'cuda'或'cpu'
        # 加载预训练的CLIP模型，使用ViT作为视觉编码器
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",
                                                                        cache_dir=cache_dir).to(device, torch.float16)
        # 加载CLIP模型的处理器，负责将输入图像转换为模型可接受的格式
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
        self.clip_model.eval()  # 设置模型为评估模式，关闭dropout等训练时特有的行为

    @torch.no_grad()  # 关闭梯度计算，推理时不需要计算梯度
    def __call__(self, img1, img2):
        # img1, img2: 输入图像，尺寸为(w, h, 3)，像素值范围为0~255
        # 返回CLIP相似度分数（余弦相似度）

        # 预处理图像，将其转换为模型所需的格式（张量），并转移到指定设备
        img1 = self.clip_processor(images=img1, return_tensors="pt")['pixel_values'].to(self.device, torch.float16)
        img2 = self.clip_processor(images=img2, return_tensors="pt")['pixel_values'].to(self.device, torch.float16)

        # 将图像传入CLIP模型，获取图像的嵌入（特征向量）
        img1_features = self.clip_model(img1).image_embeds.float()
        img2_features = self.clip_model(img2).image_embeds.float()

        # 计算图像之间的余弦相似度，得到相似度分数
        return F.cosine_similarity(img1_features, img2_features, dim=-1).item()


# 计算N-way top-k准确率
def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    # pred: 模型预测的输出（可能是概率或logits）
    # class_id: 真实类别标签
    # n_way: 在n_way分类中，取top-k的正确率
    # num_trials: 随机抽取多少次进行实验
    # top_k: 正确答案是否在前k个预测中

    if isinstance(class_id, int):  # 如果class_id是单个类别，转换为列表
        class_id = [class_id]

    pick_range = [i for i in np.arange(len(pred)) if i not in class_id]  # 获取除真实类别外的其他类别
    corrects = 0  # 统计正确的次数
    for t in range(num_trials):  # 进行多次试验
        # 随机选择n_way-1个类别
        idxs_picked = np.random.choice(pick_range, n_way - 1, replace=False)
        for gt_id in class_id:  # 对于每个真实类别
            pred_picked = torch.cat([pred[gt_id].unsqueeze(0), pred[idxs_picked]])  # 将真实类别的预测与其他类别拼接
            pred_picked = pred_picked.argsort(descending=False)[-top_k:]  # 对预测结果排序，取出top_k
            if 0 in pred_picked:  # 如果真实类别在top_k中
                corrects += 1  # 计数正确
                break
    # 返回正确率和标准差
    return corrects / num_trials, np.sqrt(corrects / num_trials * (1 - corrects / num_trials) / num_trials)


# 图像分类指标计算
@torch.no_grad()
def img_classify_metric(
        pred_videos: np.array,  # 预测的视频或图像，格式为n x H x W x C
        gt_videos: np.array,  # 真实的视频或图像
        n_way: int = 50,  # n-way分类任务
        num_trials: int = 100,  # 试验次数
        top_k: int = 1,  # top-k准确率
        cache_dir: str = '.cache',  # 缓存目录
        device: Optional[str] = 'cuda',  # 设备
        return_std: bool = False  # 是否返回标准差
):
    assert n_way > top_k  # 确保n_way大于top_k
    # 加载预训练的ViT（Vision Transformer）处理器和分类模型
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224',
                                                  cache_dir=cache_dir)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                      cache_dir=cache_dir).to(device, torch.float16)
    model.eval()  # 设置为评估模式

    acc_list = []  # 用于保存每个样本的准确率
    std_list = []  # 用于保存每个样本的标准差
    for pred, gt in zip(pred_videos, gt_videos):  # 遍历每一对预测和真实视频/图像
        # 对预测和真实图像进行处理
        pred = processor(images=pred.astype(np.uint8), return_tensors='pt')
        gt = processor(images=gt.astype(np.uint8), return_tensors='pt')

        # 对真实图像进行分类，获取top-3类别的索引
        gt_class_id = model(**gt.to(device, torch.float16)).logits.argsort(-1, descending=False).detach().flatten()[-3:]

        # 对预测图像进行分类，获取类别的概率分布
        pred_out = model(**pred.to(device, torch.float16)).logits.softmax(-1).detach().flatten()

        # 计算准确率和标准差
        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)

    if return_std:
        return acc_list, std_list  # 如果需要标准差，则返回准确率和标准差
    return acc_list  # 否则只返回准确率


# 视频分类指标计算
@torch.no_grad()
def video_classify_metric(
        pred_videos: np.array,  # 预测的视频数据
        gt_videos: np.array,  # 真实的视频数据
        n_way: int = 50,  # n-way分类任务
        num_trials: int = 100,  # 试验次数
        top_k: int = 1,  # top-k准确率
        num_frames: int = 6,  # 每个视频的帧数
        cache_dir: str = '.cache',  # 缓存目录
        device: Optional[str] = 'cuda',  # 设备
        return_std: bool = False  # 是否返回标准差
):
    assert n_way > top_k  # 确保n_way大于top_k
    # 加载预训练的VideoMAE模型处理器和分类模型
    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics',
                                                       cache_dir=cache_dir)
    model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics',
                                                           num_frames=num_frames,
                                                           cache_dir=cache_dir).to(device, torch.float16)
    model.eval()  # 设置为评估模式

    acc_list = []  # 用于保存每个样本的准确率
    std_list = []  # 用于保存每个样本的标准差

    for pred, gt in zip(pred_videos, gt_videos):  # 遍历每一对预测和真实视频
        # 对预测和真实视频进行处理
        pred = processor(list(pred), return_tensors='pt')
        gt = processor(list(gt), return_tensors='pt')

        # 对真实视频进行分类，获取top-3类别的索引
        gt_class_id = model(**gt.to(device, torch.float16)).logits.argsort(-1, descending=False).detach().flatten()[-3:]

        # 对预测视频进行分类，获取类别的概率分布
        pred_out = model(**pred.to(device, torch.float16)).logits.softmax(-1).detach().flatten()

        # 计算准确率和标准差
        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)

    if return_std:
        return acc_list, std_list  # 如果需要标准差，则返回准确率和标准差
    return acc_list  # 否则只返回准确率


# 定义一个函数计算N-way分类任务的相似度得分
def n_way_scores(
        pred_videos: np.array,  # 预测的视频数据，大小为 n x H x W x C（像素值范围为0 ~ 255）
        gt_videos: np.array,  # 真实的视频数据，大小为 n x H x W x C（像素值范围为0 ~ 255）
        n_way: int = 50,  # N-way分类任务中，选择的类别数量
        top_k: int = 1,  # 允许正确类别出现在前top_k个预测结果中
        num_trials: int = 10,  # 试验次数
        cache_dir: str = '.cache',  # 缓存目录
        device: Optional[str] = 'cuda',  # 设备，'cuda' 表示使用 GPU
):
    assert n_way > top_k  # 确保N-way分类中的类别数大于top-k

    clip_calculator = clip_score(device, cache_dir)  # 创建一个CLIP模型，用于计算图像间的相似度

    corrects = []  # 用于存储每个预测视频的正确率
    for idx, pred in enumerate(pred_videos):  # 遍历每个预测视频
        gt = gt_videos[idx]  # 获取对应的真实视频
        gt_score = clip_calculator(pred, gt)  # 计算预测视频和真实视频的相似度分数

        # 获取剩余的视频（排除当前预测视频）
        rest = np.stack([img for i, img in enumerate(gt_videos) if i != idx])
        correct_count = 0  # 正确计数器

        for _ in range(num_trials):  # 进行多次试验
            # 随机从剩余的视频中选取n_way-1个视频，构建n_way分类任务
            n_imgs_idx = np.random.choice(len(rest), n_way - 1, replace=False)
            n_imgs = rest[n_imgs_idx]
            score_list = [gt_score]  # 计算当前真实视频的相似度
            for comp in n_imgs:  # 对其他视频进行相似度计算
                comp_score = clip_calculator(pred, comp)
                score_list.append(comp_score)

            # 计算排名前top_k的预测结果，如果正确类别在前top_k中，则视为正确
            correct_count += 1 if 0 in np.argsort(score_list)[-top_k:] else 0

        # 将每次试验的正确率添加到corrects列表
        corrects.append(correct_count / num_trials)

    return corrects  # 返回每个视频的准确率


# 定义一个函数只计算CLIP模型的相似度分数
def clip_score_only(
        pred_videos: np.array,  # 预测的视频数据，大小为n x H x W x C
        gt_videos: np.array,  # 真实的视频数据，大小为n x H x W x C
        cache_dir: str = '.cache',  # 缓存目录
        device: Optional[str] = 'cuda',  # 设备
):
    clip_calculator = clip_score(device, cache_dir)  # 创建一个CLIP模型实例
    scores = []  # 用于存储每个视频对的相似度分数

    # 遍历每一对预测和真实视频
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(clip_calculator(pred, gt))  # 计算相似度分数并添加到scores中

    # 返回所有视频对的平均相似度分数
    return np.mean(scores)


# 定义一个函数，用于确保图像的通道放置在最后一个维度
def channel_last(img):
    if img.shape[-1] == 3:  # 如果图像的通道维度已经在最后
        return img
    if len(img.shape) == 3:  # 如果图像是3D（H x W x C）
        img = rearrange(img, 'c h w -> h w c')  # 重排列，使通道维度在最后
    elif len(img.shape) == 4:  # 如果图像是4D（例如视频：F x C x H x W）
        img = rearrange(img, 'f c h w -> f h w c')  # 重排列，使通道维度在最后
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')  # 如果形状不对，抛出异常
    return img


# 计算MSE（均方误差）得分
def mse_score_only(
        pred_videos: np.array,  # 预测视频数据
        gt_videos: np.array,  # 真实视频数据
        **kwargs
):
    scores = []  # 用于存储每个视频对的MSE分数
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(mse_metric(pred, gt))  # 对每一对视频计算MSE
    return np.mean(scores), np.std(scores)  # 返回MSE的平均值和标准差


# 计算SSIM（结构相似性）得分
def ssim_score_only(
        pred_videos: np.array,  # 预测视频数据
        gt_videos: np.array,  # 真实视频数据
        **kwargs
):
    pred_videos = channel_last(pred_videos)  # 确保通道维度在最后
    gt_videos = channel_last(gt_videos)  # 确保通道维度在最后
    scores = []  # 用于存储每个视频对的SSIM分数
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(ssim_metric(pred, gt))  # 计算SSIM分数并添加到scores中
    return np.mean(scores), np.std(scores)  # 返回SSIM的平均值和标准差


# 计算MSE损失（均方误差）
def mse_metric(img1, img2):
    return F.mse_loss(torch.FloatTensor(img1 / 255.0), torch.FloatTensor(img2 / 255.0), reduction='mean').item()


# 计算SSIM（结构相似性）
def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)  # 使用skimage库计算SSIM


# 处理重叠的视频片段，返回去重后的视频数据
def remove_overlap(
        pred_videos: np.array,  # 预测的视频数据
        gt_videos: np.array,  # 真实的视频数据
        scene_seg_list: List,  # 场景分割列表，表示每个视频的场景编号
        get_scene_seg: bool = False,  # 是否获取场景分割信息
):
    pred_list = []  # 存储去重后的预测视频
    gt_list = []  # 存储去重后的真实视频
    seg_dict = {}  # 用于记录已添加的场景
    for pred, gt, seg in zip(pred_videos, gt_videos, scene_seg_list):  # 遍历每对视频和场景
        if '-' not in seg:  # 如果场景编号不包含'-'（可能是非重叠场景）
            if get_scene_seg:  # 如果需要获取场景分割信息
                if seg not in seg_dict.keys():  # 如果场景没有被添加过
                    seg_dict[seg] = seg  # 添加场景到字典中
                    pred_list.append(pred)  # 添加预测视频
                    gt_list.append(gt)  # 添加真实视频
            else:
                pred_list.append(pred)  # 直接添加视频
                gt_list.append(gt)
    return np.stack(pred_list), np.stack(gt_list)  # 返回去重后的预测视频和真实视频


# GT_label表示标签列表，每个标签表示视频所属的类
GT_label = np.array([[23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33, 34, 4, 39, 17, 1, 26,
                      20, 29, 13, 32, 37, 2, 11, 12, 30, 31, 8, 21, 7, 10],
                     [27, 33, 22, 28, 31, 12, 38, 4, 18, 17, 35, 39, 40, 5, 24, 32, 15, 13, 2, 16, 34, 25, 19, 30, 23,
                      3, 8, 29, 7, 20, 11, 14, 37, 6, 21, 1, 10, 36, 26, 9],
                     [15, 36, 31, 1, 34, 3, 37, 12, 4, 5, 21, 24, 14, 16, 39, 20, 28, 29, 18, 32, 2, 27, 8, 19, 13, 10,
                      30, 40, 17, 26, 11, 9, 33, 25, 35, 7, 38, 22, 23, 6],
                     [16, 28, 23, 1, 39, 10, 35, 14, 19, 27, 37, 31, 5, 18, 11, 25, 29, 13, 20, 24, 7, 34, 26, 4, 40,
                      12, 8, 22, 21, 30, 17, 2, 38, 9, 3, 36, 33, 6, 32, 15],
                     [18, 29, 7, 35, 22, 19, 12, 36, 8, 15, 28, 1, 34, 23, 20, 13, 37, 9, 16, 30, 2, 33, 27, 21, 14, 38,
                      10, 17, 31, 3, 24, 39, 11, 32, 4, 25, 40, 5, 26, 6],
                     [29, 16, 1, 22, 34, 39, 24, 10, 8, 35, 27, 31, 23, 17, 2, 15, 25, 40, 3, 36, 26, 6, 14, 37, 9, 12,
                      19, 30, 5, 28, 32, 4, 13, 18, 21, 20, 7, 11, 33, 38],
                     [38, 34, 40, 10, 28, 7, 1, 37, 22, 9, 16, 5, 12, 36, 20, 30, 6, 15, 35, 2, 31, 26, 18, 24, 8, 3,
                      23, 19, 14, 13, 21, 4, 25, 11, 32, 17, 39, 29, 33, 27]
                     ])

# 计算GT标签中所有对应的索引
indices = [list(GT_label[6]).index(element) for element in range(1, 41)]
print(indices)

# 初始化存储各个评估指标的列表
video_2way_acc = []  # 视频2-way准确率
video_40way_acc = []  # 视频40-way准确率
image_2way_acc = []  # 图像2-way准确率
image_40way_acc = []  # 图像40-way准确率
image_ssim = []  # 图像SSIM得分

# 对200个视频样本进行评估
for i in range(200):
    class_id = i // 5  # 计算视频所属的类别
    video_clip_id = i % 5  # 获取视频在该类别中的编号
    gt_video_id = indices[class_id] * 5 + video_clip_id  # 计算真实视频的ID
    print("gt_video_id = ", gt_video_id)

    # 读取预测视频和真实视频
    gt_video = imageio.mimread('../data/small_video_3fps/test_video_gif/' + str(gt_video_id + 1) + '.gif')
    pred_video = imageio.mimread('./final_200_results/40_classes_eeg_' + str(i) + '.gif')

    # 重塑视频数据为6帧，大小为(6, 288, 512, 3)
    gt_video = np.concatenate(gt_video).reshape(6, 288, 512, 3)
    pred_video = np.concatenate(pred_video).reshape(6, 288, 512, 3)

    print("gt_video.shape = ", gt_video.shape)

    # 计算SSIM得分
    mean, std = ssim_score_only(gt_video, pred_video)
    print(mean, std)
    image_ssim.append(mean)  # 将每个视频的SSIM得分添加到列表

    # 图像40-way准确率评估
    acc = img_classify_metric(
        pred_videos=pred_video,
        gt_videos=gt_video,
        n_way=40,
        num_trials=100,
        top_k=1,
        cache_dir='.cache',
        device='cuda',
        return_std=False
    )
    print("image_40way_acc = ", acc)
    for acci in acc:
        image_40way_acc.append(acc)  # 将每个视频的准确率添加到列表

    # 图像2-way准确率评估
    acc = img_classify_metric(
        pred_videos=pred_video,
        gt_videos=gt_video,
        n_way=2,
        num_trials=100,
        top_k=1,
        cache_dir='.cache',
        device='cuda',
        return_std=False
    )
    print("image_2way_acc = ", acc)
    for acci in acc:
        image_2way_acc.append(acc)  # 将每个视频的准确率添加到列表

    # 视频40-way准确率评估
    gt_video = gt_video.reshape((1,) + gt_video.shape)  # 将视频数据转换为批次格式
    pred_video = pred_video.reshape((1,) + pred_video.shape)
    acc = video_classify_metric(
        pred_videos=pred_video,
        gt_videos=gt_video,
        n_way=40,
        num_trials=100,
        top_k=1,
        cache_dir='.cache',
        device='cuda',
        return_std=False
    )

    print("video_40way_acc = ", acc)
    for acci in acc:
        video_40way_acc.append(acc)  # 将每个视频的准确率添加到列表

    # 视频2-way准确率评估
    acc = video_classify_metric(
        pred_videos=pred_video,
        gt_videos=gt_video,
        n_way=2,
        num_trials=100,
        top_k=1,
        cache_dir='.cache',
        device='cuda',
        return_std=False
    )

    print("video_2way_acc = ", acc)
    for acci in acc:
        video_2way_acc.append(acc)  # 将每个视频的准确率添加到列表

# 输出最终的评估结果
print("video_2way_acc = ", np.mean(np.array(video_2way_acc)), np.std(np.array(video_2way_acc)))
print("video_40way_acc = ", np.mean(np.array(video_40way_acc)), np.std(np.array(video_40way_acc)))
print("image_2way_acc = ", np.mean(np.array(image_2way_acc)), np.std(np.array(image_2way_acc)))
print("image_40way_acc = ", np.mean(np.array(image_40way_acc)), np.std(np.array(image_40way_acc)))
print("image_ssim = ", np.mean(np.array(image_ssim)), np.std(np.array(image_ssim)))

# 将最终的评估结果保存为npy文件
np.save("./final_200_results/video_2way_acc.npy", np.array(video_2way_acc))
np.save("./final_200_results/video_40way_acc.npy", np.array(video_40way_acc))
np.save("./final_200_results/image_2way_acc.npy", np.array(image_2way_acc))
np.save("./final_200_results/image_40way_acc.npy", np.array(image_40way_acc))
np.save("./final_200_results/image_ssim.npy", np.array(image_ssim))

