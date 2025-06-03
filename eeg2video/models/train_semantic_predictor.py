import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import  matplotlib.pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR



# from EEG_preprocessing.extract_DE_PSD_features_1per1s import get_files_names_in_directory

def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

# 用于只传入文本embedding
class CLIP1(nn.Module):
    def __init__(self,input_dim):
        super(CLIP1, self).__init__()
        self.input_dim = input_dim  # 输入维度
    def forward(self, x):
        return x

input_dim = 77*768  # 假设输入维度是 77*768
models = CLIP1(input_dim)
model_file = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/semantic_predictor_f_eq.pt'
# 保存模型的 input_dim（而不是整个权重矩阵）
torch.save({'input_dim': input_dim}, model_file)
print("ok")

#自定义的CLIP模型，本质是一个多层MLP，用于将EEG信号映射到文本嵌入空间    以及语义提取器的输入
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 1000),#输入维度：310（EEG处理后的特征维度）   310=62*5应该是DE的输入
            # nn.Linear(400, 10000),  # 也改成对应我的维度400
            # nn.LayerNorm(1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 3000),
            nn.LayerNorm(3000),
            nn.ReLU(),
            nn.Linear(3000, 10000),
            nn.LayerNorm(10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.LayerNorm(10000),
            # nn.BatchNorm1d(50000),
            nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            # nn.Linear(10000, 77 * 768),#输出维度：77*768（可能对应CLIP文本编码器的输入维度）
            # nn.Linear(10000, 512)#也改成对应我的维度512
            # nn.LayerNorm(77 * 768)     #
        )
        # 对比学习投影头
        self.contrastive_proj = nn.Linear(10000, 77 * 768)
        # 新增分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(10000),
            nn.Dropout(0.3),
            nn.Linear(10000, 256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )
        self.scale = nn.Parameter(torch.tensor(1.0))  # 初始化为  范数

    def forward(self, eeg):
        # eeg = (eeg - eeg.mean(dim=0)) / (eeg.std(dim=0) + 1e-6)
        eeg_embeddings = self.mlp(eeg)* self.scale
        # 对比学习分支
        contrastive_emb = F.normalize(self.contrastive_proj(eeg_embeddings), dim=-1)
        # 分类分支
        logits = self.classifier(eeg_embeddings)

          # shape: (batch_size)
        # eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=-1)* self.scale  # 归一化后缩放到 315
        return contrastive_emb,logits  # 归一化 可学习



## 交叉熵
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)  # 计算 softmax 并取对数
    loss = (-targets * log_softmax(preds)).sum(1)  # 计算交叉熵损失
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# 计算每个位置（77个token）的MSE，再取平均
def structural_mse_loss(eeg_emb, text_emb):
    # eeg_emb: [B, 77, 768], text_emb: [B, 77, 768]
    per_position_loss = F.mse_loss(eeg_emb, text_emb, reduction='none')  # [B,77,768]
    return per_position_loss.mean(dim=[1,2]).mean()  # 先对空间维度平均，再对批次平均

# 定义一个数据集类，将 EEG 信号和文本数据进行标准化并存储
class Dataset():
    def __init__(self, eeg, text):
        scaler = preprocessing.StandardScaler().fit(eeg)  # 对 EEG 进行标准化
        eeg = scaler.transform(eeg)  # 归一化 EEG 数据

        self.eeg = eeg  # 归一化后的 EEG 数据
        self.text = text  # 文本嵌入
        self.len = eeg.shape[0]  # 样本数量


    def __len__(self):
        return self.len  #数据集的样本数

    def __getitem__(self, item):
        return self.eeg[item], self.text[item]  # 返回 EEG 和对应的文本嵌入

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
chosed_label = [i for i in range(1, 41)]              # set subset for training semantic predictor


# 310输入用DE的
if __name__ == '__main__':
    # 加载clip
    model = CLIP()
    # sub_list = get_files_names_in_directory("E:/TJ/store/EEG2Video-main/EEG_preprocessing/data/DE_1per1s")
    # for sub_name in sub_list:

    # 根据rearrange形状推断，使用segmented后的eeg
    eeg_data_path = "/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/DE_1per1s/sub1.npy.npy"  # your own data path for eeg data
    # eeg_data_path = "/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/DE_1per1s/"+sub_name  # your own data path for eeg data
    # eeg_data_path = "E:/TJ/store/EEG2Video-main/EEG_preprocessing/data/DE_1per1s/sub1.npy.npy"  # your own data path for eeg data
    # text_embedding_path = "text_embedding_77768_2.npy"  # your own data path for text embedding
    text_embedding_path = "text_embedding_masked2.npy"  # your own data path for text embedding

    # 加载 EEG 和文本嵌入数据
    eegdata = np.load(eeg_data_path)
    text_embedding = np.load(text_embedding_path)
    # text_embedding = rearrange(text_embedding,'a b c d -> (a b) c d')
    # text_embedding = text_embedding[:200]  # f3 1.MP4训
    # print(eegdata)
    print(eegdata.shape)
    EEG = []
    for i in range(6):
        indices = [list(GT_label[i]).index(element) for element in chosed_label]
        # chosed_eeg = eegdata[i][indices, :]
        chosed_eeg = eegdata[i]
        EEG.append(chosed_eeg)
    EEG = np.stack(EEG, axis=0)  # 转换为 NumPy 数组
    EEG = torch.from_numpy(EEG)  # 转换为 PyTorch 张量

    valEEG = []
    indices = [list(GT_label[i]).index(element) for element in chosed_label]
    # chosed_eeg = eegdata[i][indices, :]
    chosed_eeg = eegdata[6]
    valEEG.append(chosed_eeg)
    valEEG = np.stack(valEEG, axis=0)  # 转换为 NumPy 数组
    valEEG = torch.from_numpy(valEEG)  # 转换为 PyTorch 张量
    valEEG = rearrange(valEEG, 'a b c d e f -> (a b c) d (e f)')  # 重新排列张量维度 跟DE310对上了

    print("EEG.shape = ", EEG.shape)
    # EEG = EEG[0]  # F3 先取1.MP4训
    EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')  # 重新排列张量维度 跟DE310对上了
    # EEG = rearrange(EEG, 'b c d e f -> (b c) d (e f)')  # 重新排列张量维度 跟DE310对上了
    # EEG = rearrange(EEG, 'a b c d e  -> (a b c) d e')  # 维度对不上 改一下
    print("after arrange EEG.shape = ", EEG.shape)
    # print(EEG)
    print("text_embedding.shape = ", text_embedding.shape)

    Text = []
    for i in range(6):
        # Text.append(text_embedding[:30,...])
        # Text.append(text_embedding[:150,...])
        subtext = text_embedding[i * 200:(i + 1) * 200, ...]
        Text = Text + [subtext]
    Text = np.concatenate(Text)
    Text = np.stack(Text, axis=0)  # 训1.mp4先不要
    print("Text.shape = ", Text.shape)
    Text = torch.from_numpy(Text)

    valText = []
    subtext = text_embedding[1200:1400]
    valText = valText + [subtext]
    valText = np.concatenate(valText)
    valText = np.stack(valText, axis=0)  # 训1.mp4先不要
    print("valText.shape = ", valText.shape)
    valText = torch.from_numpy(valText)

    # Text = Text[:200]  # F3 先取1.MP4训
    # Text = torch.reshape(Text, (Text.shape[0], Text.shape[1]*Text.shape[2]))
    # Text = torch.reshape(Text, (Text.shape[0] * Text.shape[1], Text.shape[2]))  # 初始的不对 改成前面合并
    # Text = torch.reshape(Text, (Text.shape[0] * Text.shape[1],Text.shape[2] , Text.shape[3]))  # f3先不用
    EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], 310)  # 计算均值，调整尺寸
    valEEG = torch.mean(valEEG, dim=1).resize(valEEG.shape[0], 310)  # 计算均值，调整尺寸

    # EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], 400)  # 数不对 改一下
    print("EEGshape:", EEG.shape)
    print("Textshape:", Text.shape)



    # model_file = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/semantic_predictor_f6.pt'
    # model_file = 'E:/TJ/store/EEG2Video-main/EEG2Video/models/semantic_predictor_f1.pt'
    # if os.path.exists(model_file):
    #     model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['state_dict'])
    model.cuda()

    # 数据集和loader
    dataset = Dataset(EEG, Text)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = Dataset(valEEG, valText)
    val_batch_size = 64
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    epochs = 1000
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    warmup_steps = 100  # 预热步数
    total_steps = epochs * len(dataloader)  # 总训练步数
    eta_min = 1e-3  # 余弦退火的最小学习率
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
    )
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)  # 从 0 线性增长到 base_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler
        ],
        milestones=[warmup_steps]
    )

    loss_list=[]
    # train
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            eeg, text = batch
            eeg = eeg.float().cuda()
            text_embeddings = text.float().cuda()
            optimizer.zero_grad()
            eeg_embeddings = model(eeg)
            eeg_embeddings = rearrange(eeg_embeddings, 'b (c d)-> b c d', c=77,d=768)  # 调成和text_emb一样的形状
            # print("eeg mean std",eeg_embeddings.mean(dim=1), eeg_embeddings.std(dim=1))
            # print("text mean std",text_embeddings.mean(dim=1), text_embeddings.std(dim=1))
            # eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=-1) * scale  # 归一化后缩放到 315
            # text_embeddings = F.normalize(text_embeddings, p=2, dim=-1) * 30  # 归一化后缩放到 315

            # print("text检查范数 ",torch.norm(text_embeddings, p=2, dim=-1).mean())  # 检查范数
            # print("eeg检查范数 ",torch.norm(eeg_embeddings, p=2, dim=-1).mean())  # 检查范数
            # print(eeg_embeddings.shape)
            # print(text_embeddings.shape)

            loss = F.mse_loss(eeg_embeddings, text_embeddings)
            # # loss = cross_entropy(eeg_embeddings, text_embeddings, reduction='mean')
            # loss = F.cross_entropy(eeg_embeddings, text_embeddings, reduction='mean')
            # loss = structural_mse_loss(eeg_embeddings, text_embeddings)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5,  # 最大范数阈值
                norm_type=2  # 使用L2范数
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print("lr ",optimizer.param_groups[0]['lr'],"loss ",epoch_loss/len(dataloader))
        loss_list.append(epoch_loss/len(dataloader))

        if epoch % 50 == 0:
            # 验证 保存
            t_loss=0
            for i, batch in enumerate(val_dataloader):
                eeg, text = batch
                eeg = eeg.float().cuda()
                text_embeddings = text.float().cuda()
                eeg_embeddings = model(eeg)
                eeg_embeddings = rearrange(eeg_embeddings, 'b (c d)-> b c d', c=77, d=768)  # 调成和text_emb一样的形状
                # print("eeg mean std",eeg_embeddings.mean(dim=1), eeg_embeddings.std(dim=1))
                # print("text mean std",text_embeddings.mean(dim=1), text_embeddings.std(dim=1))
                # eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=-1) * scale  # 归一化后缩放到 315
                # text_embeddings = F.normalize(text_embeddings, p=2, dim=-1) * 30  # 归一化后缩放到 315

                # print("text检查范数 ",torch.norm(text_embeddings, p=2, dim=-1).mean())  # 检查范数
                # print("eeg检查范数 ",torch.norm(eeg_embeddings, p=2, dim=-1).mean())  # 检查范数
                # print(eeg_embeddings.shape)
                # print(text_embeddings.shape)

                loss = F.mse_loss(eeg_embeddings, text_embeddings)
                # # loss = cross_entropy(eeg_embeddings, text_embeddings, reduction='mean')
                # loss = F.cross_entropy(eeg_embeddings, text_embeddings, reduction='mean')
                # loss = structural_mse_loss(eeg_embeddings, text_embeddings)

                t_loss += loss.item()
            print("epoch ",epoch," val loss ",t_loss / len(val_dataloader))
            model_dict = model.state_dict()
            torch.save({'state_dict': model_dict}, 'semantic_predictor_f7.pt')


    model_dict = model.state_dict()
    torch.save({'state_dict': model_dict}, 'semantic_predictor_f7.pt')

    plt.plot(range(len(loss_list)),loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

'''
f1 确定只在sub1上训 
f2 lr增到0.05 训400 model加一层      loss16
f3 给输出eeg加归一化  1.mp4训200  lr5e3 
f4 2000lun 1.mp4 5e-2 loss 0.02（1000轮左右就不下降了）
f5 2000轮 1.mp4 5e-2退2e-2   loss0.01
f6 1000   all  5e-2退2e-2  把incidice去掉        2000 all 1e-2 50轮val 
f7 1000 换分布正确的masked2  1e-2退1e-4
'''


# #400输入用原脑电数据的
# if __name__ == '__main__':
#     # 根据rearrange形状推断，使用segmented后的eeg
#     # eeg_data_path = "/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/Segmented_Rawf_200Hz_2s/sub1.npy"               # your own data path for eeg data
#     eeg_data_path = "E:/TJ/store/EEG2Video-main/EEG_preprocessing/data/Segmented_Rawf_200Hz_2s/sub1.npy"               # your own data path for eeg data
#     # eeg_data_path = "/home/bcilab/Ljx/EEG2Video-main/SEED-DV/EEG/sub1.npy"                        # your own data path for eeg data
#     text_embedding_path = "text_embedding.npy"        # your own data path for text embedding
#
#     # 加载 EEG 和文本嵌入数据
#     eegdata = np.load(eeg_data_path)
#     text_embedding = np.load(text_embedding_path)
#     print(eegdata.shape)
#     EEG = []
#     for i in range(6):
#         indices = [list(GT_label[i]).index(element) for element in chosed_label]
#         chosed_eeg = eegdata[i][indices,:]
#         EEG.append(chosed_eeg)
#
#     EEG = np.stack(EEG, axis=0) # 转换为 NumPy 数组
#     EEG = torch.from_numpy(EEG) # 转换为 PyTorch 张量
#     print("EEG.shape = ", EEG.shape)
#     # EEG = rearrange(EEG, 'a b c d e f -> (a b c) d (e f)')  # 重新排列张量维度
#     EEG = rearrange(EEG, 'a b c d e  -> (a b c) d e')  # 维度对不上 改一下
#
#
#     print("after arrange EEG.shape = ", EEG.shape)
#
#     print(EEG)
#     print("text_embedding.shape = ", text_embedding.shape)
#
#     Text = []
#     for i in range(6):
#         # Text.append(text_embedding[:30,...])
#         # Text.append(text_embedding[:150,...])
#         subtext=text_embedding[i*200:(i+1)*200,...]
#         Text=Text + [subtext]
#
#     # Text = np.concatenate(Text)
#     Text = np.stack(Text, axis=0)
#
#     print("Text.shape = ", Text.shape)
#
#     Text = torch.from_numpy(Text)
#     # Text = torch.reshape(Text, (Text.shape[0], Text.shape[1]*Text.shape[2]))
#     Text = torch.reshape(Text, (Text.shape[0]* Text.shape[1],Text.shape[2]))   # 初始的不对 改成前面合并
#     # EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], 310)  # 计算均值，调整尺寸
#     EEG = torch.mean(EEG, dim=1).resize(EEG.shape[0], 400)  # 数不对 改一下
#     print(EEG)
#     print("EEGshape:",EEG.shape)
#     print("Textshape:",Text.shape)
#
#     # 加载clip
#     model = CLIP()
#     # 没有pt先不加载
#     model_file = '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/semantic_predictor_400.pt'
#     if os.path.exists(model_file):
#         model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['state_dict'])
#         #通过将 'state_dict' 中的内容传递给它，load_state_dict() 会将权重正确加载到当前的模型。
#     model.cuda()
#
#     #数据集和loader
#     dataset = Dataset(EEG, Text)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(dataloader))
#
#     # train
#     for epoch in tqdm(range(200)):
#         model.train()
#         epoch_loss = 0
#         for i, batch in enumerate(dataloader):
#             eeg, text = batch
#             eeg = eeg.float().cuda()
#             text_embeddings = text.float().cuda()
#             optimizer.zero_grad()
#             eeg_embeddings = model(eeg)
#
#             loss = F.mse_loss(eeg_embeddings, text_embeddings)
#
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             epoch_loss += loss.item()
#         print(epoch_loss)
#
#     model_dict = model.state_dict()
#     torch.save({'state_dict': model_dict}, 'semantic_predictor_400.pt')
