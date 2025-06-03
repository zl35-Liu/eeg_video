import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskEEGConvNet(nn.Module):
    def __init__(self, task_config,input_dim=1, proj_dim=256):
        """
        task_config: 字典 {任务名: 类别数量}
        """
        super().__init__()
        # 共享的特征提取器
        self.shared_encoder = nn.Sequential(
            # Block 1: [B,1,62,5] → [B,32,62,2]
            nn.Conv2d(input_dim, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 时间维度5→2

            # Block 2: [B,32,62,2] → [B,64,31,2]（关键修正）
            nn.Conv2d(32, 64, kernel_size=(3, 1),  # 时间维度kernel改为1
                      stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 3: [B,64,31,2] → [B,512,16,1]
            nn.Conv2d(64, 512, kernel_size=(3, 2),
                      stride=(2, 2), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.GELU(),

            # Block4: [B,512,16,1] → [B,1024,16,1]
            nn.Conv2d(512, 1024, kernel_size=(3, 1),
                      padding=(1, 0)),
            nn.BatchNorm2d(1024),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # 输出[B,1024]
        )

        # 对比学习投影头
        self.contrastive_proj = nn.Sequential(
            nn.Linear(1024, 512),  # 降维而非升维
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),  # 提高Dropout比例

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(256, 77 * 768)  # 最终维度保持相同但路径更稳定
        )

        # 为每个任务创建独立的分类头
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_config.items():
            self.classifiers[task_name] = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        # 可学习缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg):
        # 输入重塑: (B,310) → (B,1,62,5)
        eeg = eeg.view(-1, 1, 62, 5)
        shared_features = self.shared_encoder(eeg)
        eeg_emb = F.normalize(self.contrastive_proj(shared_features), dim=-1) * self.scale

        # 计算每个任务的logits
        task_logits = {}
        for task_name, classifier in self.classifiers.items():
            task_logits[task_name] = classifier(shared_features)

        return eeg_emb, task_logits

# 改进enco  原本mlp参数太大会过拟合 并且没有时空卷积
class EEGConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=40, clip_dim=77 * 768):
        super(EEGConvNet, self).__init__()
        # 时空特征提取主干（修正卷积参数）
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv1d(1, 32, kernel_size=15, padding=7),  # 捕捉长时序依赖
        #     nn.BatchNorm1d(32),
        #     nn.GELU(),
        #     nn.MaxPool1d(2),
        #
        #     nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm1d(64),
        #     nn.GELU(),
        #
        #     nn.Conv1d(64, 512, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten()  # 输出[B,128]
        # )
        self.feature_extractor = nn.Sequential(
            # Block 1: [B,1,62,5] → [B,32,62,2]
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 时间维度5→2

            # Block 2: [B,32,62,2] → [B,64,31,2]（关键修正）
            nn.Conv2d(32, 64, kernel_size=(3, 1),  # 时间维度kernel改为1
                      stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Block 3: [B,64,31,2] → [B,512,16,1]
            nn.Conv2d(64, 512, kernel_size=(3, 2),
                      stride=(2, 2), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.GELU(),

            # Block4: [B,512,16,1] → [B,1024,16,1]
            nn.Conv2d(512, 1024, kernel_size=(3, 1),
                      padding=(1, 0)),
            nn.BatchNorm2d(1024),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # 输出[B,1024]
        )

        # 动态特征维度计算
        self.demo_net = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        # 改进方案：渐进式投影
        self.contrastive_proj = nn.Sequential(
            nn.Linear(1024, 512),  # 降维而非升维
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),  # 提高Dropout比例

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(256, 77 * 768)  # 最终维度保持相同但路径更稳定
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # 添加BatchNorm
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 可学习缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 输入重塑: (B,310) → (B,1,62,5)
        x = x.view(-1, 1, 62, 5)

        features = self.feature_extractor(x)  # (B,1024)
        proj_features = self.demo_net(features)  # (B,1024)

        contrastive_emb = F.normalize(
            self.contrastive_proj(proj_features), dim=-1) * self.scale
        logits = self.classifier(proj_features)

        return contrastive_emb, logits


# class EEGEncoder(nn.Module):
#     def __init__(self, input_channels=2, proj_dim=128):
#         super().__init__()
#         # 时空特征提取
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#
#             nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 2))
#         )
#
#         # 全连接投影
#         self.projection = nn.Sequential(
#             nn.Linear(128 * 4 * 2, 512),
#             nn.ReLU(),
#             nn.Linear(512, proj_dim)
#         )
#
#     def forward(self, x):
#         x = self.conv_block(x)
#         x = x.flatten(1)
#         return self.projection(x)


# class CLIPTextEncoder(nn.Module):
#     def __init__(self, model_type="ViT-B/32"):
#         super().__init__()
#         self.clip_model = clip.load(model_type)[0]
#         self.tokenizer = clip.tokenize
#
#         # 冻结CLIP参数
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#     def forward(self, text):
#         with torch.no_grad():
#             text_features = self.clip_model.encode_text(text)
#         return text_features.float()


# class ContrastiveModel(nn.Module):
#     def __init__(self, proj_dim=256):
#         super().__init__()
#         self.eeg_encoder = EEGEncoder(proj_dim=proj_dim)
#         # self.text_encoder = CLIPTextEncoder()  # 使用之前定义的CLIP文本编码器
#
#         # 增强的投影头
#         self.eeg_proj = nn.Sequential(
#             nn.LayerNorm(proj_dim),
#             nn.Linear(proj_dim, proj_dim * 2),
#             nn.GELU(),
#             nn.Linear(proj_dim * 2, proj_dim)
#         )
#         self.text_proj = nn.Sequential(
#             nn.LayerNorm(512),  # CLIP文本特征维度
#             nn.Linear(512, proj_dim * 2),
#             nn.GELU(),
#             nn.Linear(proj_dim * 2, proj_dim)
#         )
#
#     def forward(self, eeg, text):
#         eeg_feat = F.normalize(self.eeg_proj(self.eeg_encoder(eeg)), dim=-1)
#         text_feat = F.normalize(self.text_proj(self.text_encoder(text)), dim=-1)
#         return eeg_feat, text_feat