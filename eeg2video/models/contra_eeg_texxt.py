import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
from tqdm import tqdm

# 固定所有随机种子
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib.pyplot as plt

from train_semantic_predictor import CLIP
from model_contrastive import MultiTaskEEGConvNet,EEGConvNet
from utils import (
    in_batch_contrastive_loss, multitask_loss, multitasks_loss,
    calculate_topk_accuracy, plot_loss_curves, compute_global_mean_var, plot_accuracy_curves
)
print("hello")



class EEGTextDataset(Dataset):
    def __init__(self, eeg_data, text_emb, class_labels_dict):
        """
        eeg_data: tensor of shape (N, 2, 62, 5)
        text_descriptions: list of N strings
        class_labels_dict: dictionary mapping video names to class labels
        """
        self.eeg_data = eeg_data
        self.text_emb = text_emb
        self.labels_dict = class_labels_dict  # 直接使用现有数组

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        labels = {task: torch.tensor(labels[idx]) for task, labels in self.labels_dict.items()}  # 字典中取出 对应某种分类的标签
        return self.eeg_data[idx], self.text_emb[idx],labels



# 增强的训练流程
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    eeg_data_path = "/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/DE_1per1s/sub1.npy.npy"  # sub1 7个视频
    text_embedding_path = "/home/bcilab/Ljx/EEG2Video-main/EEG2Video/models/text_embedding_masked2.npy"  # all文本
    eegdata = np.load(eeg_data_path)
    text_embedding = np.load(text_embedding_path)

    # 对EEG数据计算
    eeg_mean, eeg_var = compute_global_mean_var(eegdata)
    print("EEG总体均值:", eeg_mean)
    print("EEG总体方差:", eeg_var)
    eegdata = (eegdata - eeg_mean) / np.sqrt(eeg_var)
    eeg_mean1, eeg_var1 = compute_global_mean_var(eegdata)
    print("EEG改后均值:", eeg_mean1)
    print("EEG改后方差:", eeg_var1)
    # 对文本embedding计算
    text_mean, text_var = compute_global_mean_var(text_embedding)
    print("文本embedding总体均值:", text_mean)
    print("文本embedding总体方差:", text_var)

    eeg_val = eegdata[6]
    eeg_val = rearrange(eeg_val, 'b c d e f -> (b c) d (e f)')
    eeg_val = torch.from_numpy(eeg_val).float()  # 转换为 PyTorch 张量
    eeg_val = torch.mean(eeg_val, dim=1).resize(eeg_val.shape[0], 310)  # 计算均值，调整尺寸
    eegdata = eegdata[:6]
    eegdata = rearrange(eegdata, 'a b c d e f -> (a b c) d (e f)')  # 重新排列张量维度 跟DE310对上了
    eegdata = torch.from_numpy(eegdata).float()  # 转换为 PyTorch 张量
    eegdata = torch.mean(eegdata, dim=1).resize(eegdata.shape[0], 310)  # 计算均值，调整尺寸
    print(eegdata.shape)
    print(f"NaN比例：{torch.isnan(eegdata).sum() / eegdata.numel():.2%}")   # 验证mean resize是否影响数据

    text_val = text_embedding[1200:]
    text_val = torch.from_numpy(text_val).float()  # 转换为 PyTorch 张量
    text_val= torch.reshape(text_val, (text_val.shape[0], text_val.shape[1]*text_val.shape[2]))
    text_embedding = text_embedding[:1200]
    text_embedding = torch.from_numpy(text_embedding).float()
    text_embedding= torch.reshape(text_embedding, (text_embedding.shape[0], text_embedding.shape[1]*text_embedding.shape[2]))
    print(text_embedding.shape)

    GT_label = np.array([[23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33,
                          34, 4, 39, 17, 1, 26, 20, 29, 13, 32, 37, 2, 11, 12, 30, 31, 8, 21, 7, 10, ],
                         [27, 33, 22, 28, 31, 12, 38, 4, 18, 17, 35, 39, 40, 5, 24, 32, 15, 13, 2, 16,
                          34, 25, 19, 30, 23, 3, 8, 29, 7, 20, 11, 14, 37, 6, 21, 1, 10, 36, 26, 9, ],
                         [15, 36, 31, 1, 34, 3, 37, 12, 4, 5, 21, 24, 14, 16, 39, 20, 28, 29, 18, 32,
                          2, 27, 8, 19, 13, 10, 30, 40, 17, 26, 11, 9, 33, 25, 35, 7, 38, 22, 23, 6, ],
                         [16, 28, 23, 1, 39, 10, 35, 14, 19, 27, 37, 31, 5, 18, 11, 25, 29, 13, 20, 24,
                          7, 34, 26, 4, 40, 12, 8, 22, 21, 30, 17, 2, 38, 9, 3, 36, 33, 6, 32, 15, ],
                         [18, 29, 7, 35, 22, 19, 12, 36, 8, 15, 28, 1, 34, 23, 20, 13, 37, 9, 16, 30,
                          2, 33, 27, 21, 14, 38, 10, 17, 31, 3, 24, 39, 11, 32, 4, 25, 40, 5, 26, 6, ],
                         [29, 16, 1, 22, 34, 39, 24, 10, 8, 35, 27, 31, 23, 17, 2, 15, 25, 40, 3, 36,
                          26, 6, 14, 37, 9, 12, 19, 30, 5, 28, 32, 4, 13, 18, 21, 20, 7, 11, 33, 38],
                         [38, 34, 40, 10, 28, 7, 1, 37, 22, 9, 16, 5, 12, 36, 20, 30, 6, 15, 35, 2,
                          31, 26, 18, 24, 8, 3, 23, 19, 14, 13, 21, 4, 25, 11, 32, 17, 39, 29, 33, 27]
                         ])
    All_label = np.empty((0, 200))
    for block_id in range(7):
        All_label = np.concatenate((All_label, GT_label[block_id].repeat(5).reshape(1, 200)))  #
    # print("class label shape",All_label.shape)
    class_labels = rearrange(All_label, 'b c -> (b c)') - 1
    class_labels_val = class_labels[1200:]
    class_labels = class_labels[:1200]
    # print("class label shape", class_labels.shape)

    # 标签-其他
    GT_label1 = np.load('/home/bcilab/Ljx/EEG2Video-main/dataset/meta_info/All_video_face_apperance.npy')   # 初始将list转np 尝试直接np load
    GT_label1 = rearrange(GT_label1, 'b c -> (b c)')
    GT_label1_val = GT_label1[1200:]
    GT_label1 = GT_label1[:1200]
    GT_label2 = np.load('/home/bcilab/Ljx/EEG2Video-main/dataset/meta_info/All_video_human_apperance.npy')   # 初始将list转np 尝试直接np load
    GT_label2 = rearrange(GT_label2, 'b c -> (b c)')
    GT_label2_val = GT_label2[1200:]
    GT_label2 = GT_label2[:1200]
    GT_label3 = np.load('/home/bcilab/Ljx/EEG2Video-main/dataset/meta_info/All_video_obj_number.npy')   # 初始将list转np 尝试直接np load
    GT_label3 = rearrange(GT_label3, 'b c -> (b c)')-1
    GT_label3_val = GT_label3[1200:]
    GT_label3 = GT_label3[:1200]
    GT_label4 = np.load('/home/bcilab/Ljx/EEG2Video-main/dataset/meta_info/All_video_optical_flow_score.npy')   # 初始将list转np 尝试直接np load
    GT_label4 = np.where(GT_label4 >1.799,1,0)
    GT_label4= GT_label4.astype(np.int64)
    GT_label4 = rearrange(GT_label4, 'b c -> (b c)')
    GT_label4_val = GT_label4[1200:]
    GT_label4 = GT_label4[:1200]
    # print(GT_label4.shape)


    # 定义多个分类任务
    task_config = {
        "video_category": 40,  # 视频类别任务，40类
        "face_appearance": 2,  # 人脸出现任务，2类
        "human_appearance": 2,  # 人出现任务，2类
        "object_count": 3,  # 物体数量任务，3类
        "flow": 2,  # 快慢任务，2类
    }
    # 定义任务权重
    loss_weights = {
        "video_category":   0.3,
        "face_appearance":  0.2,
        "human_appearance": 0.2,
        "object_count":     0.1,
        "flow":             0.05,
    }
    # 加载所有标签数据
    labels_dict = {
        "video_category": class_labels,  # 您的原始标签
        "face_appearance": GT_label1,
        "human_appearance": GT_label2,
        "object_count": GT_label3,
        "flow": GT_label4,
    }
    labels_dict_val = {
        "video_category": class_labels_val,  # 您的原始标签
        "face_appearance": GT_label1_val,
        "human_appearance": GT_label2_val,
        "object_count": GT_label3_val,
        "flow": GT_label4_val,
    }


    # 假设有预处理好的数据
    # dataset = EEGTextDataset(eegdata, text_embedding,torch.from_numpy(class_labels[:1200]).long())
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # val_dataset = EEGTextDataset(eeg_val,text_val,torch.from_numpy(class_labels[1200:]).long())
    # val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    dataset = EEGTextDataset(eegdata, text_embedding,labels_dict)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    val_dataset = EEGTextDataset(eeg_val,text_val,labels_dict_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # model = ContrastiveModel(proj_dim=256).to(device)
    # model = CLIP().to(device)
    # model = EEGConvNet(num_classes=40).to(device)
    model = MultiTaskEEGConvNet(task_config).to(device)
    epochs=200

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6, weight_decay=0.05)
    print(type(optimizer.param_groups))
    print(optimizer.param_groups[0]['lr'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device='cuda', enabled=True)  # 新版本API
    # autocast = torch.amp.autocast(device_type='cuda', dtype=torch.float16)  # 新版本API

    sample_eeg, sample_text,lasbels = next(iter(dataloader))
    print(f"EEG输入范围: [{sample_eeg.min():.2f}, {sample_eeg.max():.2f}]")  # 应≈[-3,3]
    print(f"文本输入范围: [{sample_text.min():.2f}, {sample_text.max():.2f}]")  # 应≈[-3,3]
    print(f"文本嵌入范数: {sample_text.norm(dim=-1).mean():.2f}")  # 应≈1.0


    loss_list = []
    val_list = []
    train_acc_list = {}
    for task_name in task_config.keys():
        train_acc_list[task_name] = []
    train_top5_list = []
    val_acc_list = {}
    for task_name in task_config.keys():
        val_acc_list[task_name] = []
    val_top5_list = []
    all_val_preds = [] #预测标签
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        cls_correct = {}  # !!! 添加分类正确计数器
        top5_correct = {}  # 新增Top5计数器
        for task_name in task_config.keys():
            cls_correct[task_name] = 0     # 初始化每一类都 0
        for task_name in task_config.keys():
            top5_correct[task_name] = 0     # 初始化每一类都 0

        for eeg_batch, text_batch, labels in dataloader:
            # 转换标签格式
            labels_dict_batch = {}
            for task in task_config.keys():
                labels_dict_batch[task] = labels[task].to(device).long()
            eeg_batch = eeg_batch.to(device)    # 尝试不使用dtype=torch.float16 让autocast管理
            text_batch = text_batch.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                eeg_emb, cls_logits = model(eeg_batch)
                # print(eeg_emb.shape)
                # loss = in_batch_contrastive_loss(eeg_emb, text_batch)
                # total_loss_train, cont_loss, cls_loss = multitask_loss(  # !!! 多任务损失
                #     eeg_emb, text_batch, cls_logits, labels
                # )
                total_loss_train, cont_loss, cls_loss = multitasks_loss(  # !!! 多任务损失  多种分类版
                    eeg_emb, text_batch, cls_logits, labels_dict_batch, loss_weights
                )

            # loss.backward()
            # optimizer.step()
            scaler.scale(total_loss_train).backward()    # 使用总损失
            scaler.step(optimizer)
            scaler.update()

            # 计算分类准确率
            # preds = torch.argmax(cls_logits.detach(), dim=1)  # !!!
            # cls_correct += (preds == labels).sum().item()
            # top5_correct += calculate_topk_accuracy(cls_logits.detach(), labels, k=5) * labels.size(0)
            total_loss += total_loss_train.item()
            # # total_loss += loss.item()
            # 计算各任务准确率
            task_acc = {}
            for task_name in cls_logits.keys():
                preds = torch.argmax(cls_logits[task_name], dim=1)
                correct = (preds == labels_dict_batch[task_name]).sum().item()
                cls_correct[task_name] += correct
                task_acc[task_name] = correct / len(preds)

                # top5_correct += calculate_topk_accuracy(cls_logits[task_name].detach(), labels, k=5) * labels.size(0)

            # 打印各任务准确率
            acc_str = " | ".join([f"{task}: {acc:.2%}" for task, acc in task_acc.items()])
            # print(f"Epoch {epoch} | Acc: {acc_str}")

        # 计算并记录训练准确率
        for task_name in cls_logits.keys():
            epoch_acc = cls_correct[task_name] / len(dataset)
            train_acc_list[task_name].append(epoch_acc)

        # epoch_top5 = top5_correct / len(dataset)
        # train_top5_list.append(epoch_top5)

        # ======= 验证阶段 ======= +++
        if (epoch + 1) % 5 == 0:  # 每5个epoch验证一次
            model.eval()
            val_loss = 0
            val_correct = {}
            for task_name in cls_logits.keys():
                val_correct[task_name] = 0
            val_top5 = 0
            epoch_preds = []  # 当前epoch的预测

            with torch.no_grad():
                for eeg_val, text_val, labels_val in val_loader:
                    eeg_val = eeg_val.to(device)
                    text_val = text_val.to(device)
                    labels_dict_batch = {}
                    for task in task_config.keys():
                        labels_dict_batch[task] = labels_val[task].to(device).long()

                    # 前向传播
                    eeg_emb, cls_logits = model(eeg_val)
                    # cont_loss = in_batch_contrastive_loss(eeg_emb, text_val)
                    # total_loss_train, _, cls_loss = multitask_loss(
                    #     eeg_emb, text_val, cls_logits, labels_val
                    # )
                    total_loss_train, cont_loss, cls_loss = multitasks_loss(  # !!! 多任务损失  多种分类版
                        eeg_emb, text_val, cls_logits, labels_dict_batch, loss_weights
                    )

                    # preds = torch.argmax(cls_logits, dim=1)  # 预测值
                    # epoch_preds.extend(preds.cpu().numpy().tolist())
                    # # 累计指标
                    val_loss += total_loss_train.item()
                    # preds = torch.argmax(cls_logits, dim=1)
                    # val_correct += (preds == labels_val).sum().item()
                    # val_top5 += calculate_topk_accuracy(cls_logits, labels_val, k=5) * labels_val.size(0)
                    task_acc = {}
                    for task_name in cls_logits.keys():
                        preds = torch.argmax(cls_logits[task_name], dim=1)
                        correct = (preds == labels_dict_batch[task_name]).sum().item()
                        val_correct[task_name] += correct
                        task_acc[task_name] = correct / len(preds)
                    # 打印各任务准确率
                    acc_str = " val ".join([f"{task}: {acc:.2%}" for task, acc in task_acc.items()])
                    # print(f"Epoch {epoch} | Val_Acc: {acc_str}")

            # 计算验证指标
            avg_val_loss = val_loss / len(val_loader)
            val_list.append(avg_val_loss)
            for task_name in cls_logits.keys():
                epoch_acc = val_correct[task_name] / len(val_dataset)
                val_acc_list[task_name].append(epoch_acc)
            # val_top5_acc = val_top5 / len(val_dataset)
            # val_top5_list.append(val_top5_acc)

            # 保存最佳模型
            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     torch.save(model.state_dict(), "best_model.pth")
            #     print(f"保存最佳模型，验证损失：{avg_val_loss:.4f}")

            acc_str = " | ".join([f"{task}: {acc[-1]:.2%}" for task, acc in val_acc_list.items()])
            print(f"验证结果 | Loss: {avg_val_loss:.4f} {acc_str}")
            #       f"Top1: {val_acc:.2%} Top5: {val_top5_acc:.2%}")
            if epoch == epochs -1:
                print("pred",epoch_preds)
                print("label",labels_val)

        lr_scheduler.step()
        # epoch_acc = cls_correct / len(dataset)
        # top5_acc = top5_correct / len(dataset)
        loss_list.append(total_loss / len(dataloader))
        current_lr = optimizer.param_groups[0]['lr']
        print(optimizer.param_groups[0]['lr'])
        # print(optimizer.param_groups[0]['lr'])
        # print(type(epoch))  # 应为 int
        # print(type(optimizer.param_groups[0]['lr']))  # 应为 float
        # print(type(total_loss))  # 应为 float
        # print(type(cont_loss))  # 应为 float
        # print(type(cls_loss))  # 应为 float
        # print(f"Epoch {epoch} | lr {current_lr} | Loss: {total_loss / len(dataloader):.4f}"
        #       f"Cont: {cont_loss:.4f} Cls: {cls_loss:.4f} "  # !!!
        #       f"top1: {epoch_acc:.2%} top5: {top5_acc:.2%}")  # !!!
        acc_str = " | ".join([f"{task}: {acc[-1]:.2%}" for task, acc in train_acc_list.items()])
        print(f"Epoch {epoch} | lr {current_lr} | Loss: {total_loss / len(dataloader):.4f} {acc_str}"
              )  # !!!
        # print(f"Epoch {epoch} | lr {optimizer.param_groups[0]['lr']} | Avg Loss: {total_loss / len(dataloader):.4f}")
    plot_loss_curves(loss_list, val_list)
    for task_name in labels_dict.keys():
        # print(task_name,train_acc_list[task_name])
        plot_accuracy_curves(task_name,train_acc_list[task_name], val_acc_list[task_name])
        # plt.plot(train_acc_list[task_name], label=f"{task_name} Train Acc")
        # plt.plot(val_acc_list[task_name], label=f"{task_name} Val Acc")


if __name__ == "__main__":
    train()
