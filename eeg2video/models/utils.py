import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def in_batch_contrastive_loss(eeg_emb, text_emb, temperature=0.1):
    """
    计算batch内正负样本对比损失
    假设每个样本i的eeg_i对应text_i是正样本对
    """
    # 计算相似度矩阵
    logits = torch.matmul(eeg_emb, text_emb.T) / temperature  # (B,B)

    # 构造标签：对角线位置为正样本
    labels = torch.arange(logits.size(0), device=eeg_emb.device)

    # 双向对比损失
    loss_eeg = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels)
    return (loss_eeg + loss_text) / 2


    # logits = torch.matmul(eeg_emb, text_emb.T) / temperature
    #
    # # 数值保护
    # logits = torch.clamp(logits, min=-50.0, max=50.0)  # 防止exp溢出
    # logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # 数值稳定化
    #
    # labels = torch.arange(len(logits), device=logits.device)
    # return F.cross_entropy(logits, labels, label_smoothing=0.1)  # 添加标签平滑

# 加入分类 多任务损失
def multitask_loss(eeg_emb, text_emb, logits, labels, alpha=0.1):
    # 对比损失
    contrastive_loss = in_batch_contrastive_loss(eeg_emb, text_emb)

    # 分类损失
    cls_loss = F.cross_entropy(logits, labels)

    # 加权求和
    total_loss = (1 - alpha) * contrastive_loss + alpha * cls_loss
    return total_loss, contrastive_loss, cls_loss

# 为了多个分类loss 合起来
def multitasks_loss(eeg_emb, text_emb, task_logits, labels_dict, loss_weights):
    """
    eeg_emb: EEG嵌入
    text_emb: 文本嵌入
    task_logits: 各任务的logits字典 {任务名: logits}
    labels_dict: 各任务的标签字典 {任务名: 标签}
    loss_weights: 各任务的权重字典 {任务名: 权重}
    """
    # 对比损失
    contrastive_loss = in_batch_contrastive_loss(eeg_emb, text_emb)

    # 各分类任务的损失
    cls_losses = {}
    total_cls_loss = 0
    for task_name in task_logits.keys():
        cls_loss = F.cross_entropy(task_logits[task_name], labels_dict[task_name])
        cls_losses[task_name] = cls_loss
        total_cls_loss += loss_weights[task_name] * cls_loss

    # 总损失 = 对比损失 * (1 - sum(权重)) + 各分类损失加权和
    alpha_sum = sum(loss_weights.values())
    total_loss = (1 - alpha_sum) * contrastive_loss + total_cls_loss

    return total_loss, contrastive_loss, cls_losses

def calculate_topk_accuracy(logits, labels, k=5):
    """
    计算TopK准确率（修正版）
    """
    # 获取topk预测索引 [batch_size, k]
    _, topk_preds = torch.topk(logits, k, dim=1)

    # 扩展labels维度用于广播比较 [batch_size, 1] => [batch_size, k]
    labels = labels.view(-1, 1).expand(-1, k)

    # 计算正确预测数（标量）
    correct = torch.eq(topk_preds, labels).sum().float()
    return correct.item() / labels.size(0)

def plot_loss_curves(loss_list, val_list, val_interval=5, save_path="loss_curve.png"):
    """
    参数说明:
    loss_list: 每个epoch的训练损失列表
    val_list: 每个验证点的验证损失列表
    val_interval: 验证间隔的epoch数（默认5）
    save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))

    # 绘制训练损失曲线
    plt.plot(loss_list, 'b-', label='Training Loss')

    # 计算验证点的x轴位置
    x_val = [val_interval * (i + 1) - 1 for i in range(len(val_list))]  # 假设从0开始计数

    # 绘制验证损失曲线（带标记点）
    plt.plot(x_val, val_list, 'ro-', markersize=4, linewidth=1, label='Validation Loss')

    # 图表装饰
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 自动调整x轴刻度
    max_epoch = len(loss_list)
    plt.xticks(
        ticks=range(0, max_epoch, max(1, max_epoch // 10)),
        labels=[i + 1 for i in range(0, max_epoch, max(1, max_epoch // 10))]
    )

    # 保存并显示
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# plot
def plot_accuracy_curves(task_name,train_acc_list, val_acc_list, val_interval=5, save_path="accuracy_curve.png"):
    """
    新增的准确率曲线绘制函数
    参数说明:
    train_acc_list: 每个epoch的训练准确率列表
    val_acc_list: 每个验证点的验证准确率列表
    val_interval: 验证间隔的epoch数（默认5）
    save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))

    # 绘制训练准确率曲线
    plt.plot(train_acc_list, 'g-', label='Training Accuracy')

    # 计算验证点的x轴位置（与验证损失对齐）
    x_val = [val_interval * (i + 1) - 1 for i in range(len(val_acc_list))]

    # 绘制验证准确率曲线（带标记点）
    plt.plot(x_val, val_acc_list, 'mo-', markersize=4, linewidth=1, label='Validation Accuracy')

    # 图表装饰
    plt.title(f"{task_name} Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 自动调整x轴刻度
    max_epoch = len(train_acc_list)
    plt.xticks(
        ticks=range(0, max_epoch, max(1, max_epoch // 10)),
        labels=[i + 1 for i in range(0, max_epoch, max(1, max_epoch // 10))]
    )

    # 保存并显示
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_curves_1and5(train_acc_list, val_acc_list,
                         train_top5_list=None, val_top5_list=None,
                         val_interval=5, save_path="accuracy_curve.png"):
    plt.figure(figsize=(12, 6))

    # ====== 新增维度验证 ======
    if train_top5_list is not None:
        assert len(train_top5_list) == len(train_acc_list), \
            f"训练数据维度不匹配: Top1({len(train_acc_list)}) vs Top5({len(train_top5_list)})"

    if val_top5_list is not None:
        assert len(val_top5_list) == len(val_acc_list), \
            f"验证数据维度不匹配: Top1({len(val_acc_list)}) vs Top5({len(val_top5_list)})"

    # ====== 训练曲线 ======
    x_train = range(len(train_acc_list))
    plt.plot(x_train, train_acc_list, 'g-', label='Training Top1')

    if train_top5_list:
        plt.plot(x_train, train_top5_list, 'g--', alpha=0.5, label='Training Top5')

    # ====== 验证曲线 ======
    if val_acc_list:
        x_val = [val_interval * (i + 1) - 1 for i in range(len(val_acc_list))]
        plt.plot(x_val, val_acc_list, 'mo-', markersize=4, label='Validation Top1')

        if val_top5_list:
            plt.plot(x_val, val_top5_list, 'm--', markersize=4, label='Validation Top5')

    # ====== 动态调整坐标轴 ======
    all_values = []
    all_values.extend(train_acc_list)
    if train_top5_list: all_values.extend(train_top5_list)
    if val_acc_list: all_values.extend(val_acc_list)
    if val_top5_list: all_values.extend(val_top5_list)

    y_padding = 0.05  # 5%边距
    y_min = max(0.0, min(all_values) - y_padding)
    y_max = min(1.0, max(all_values) + y_padding)
    plt.ylim(y_min, y_max)

    # ====== 图表装饰 ======
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 自动调整x轴刻度
    max_epoch = len(train_acc_list)
    plt.xticks(
        ticks=range(0, max_epoch, max(1, max_epoch // 10)),
        labels=[i + 1 for i in range(0, max_epoch, max(1, max_epoch // 10))]
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_global_mean_var(data):
    flattened_data = data.flatten()  # 展平为1D数组
    global_mean = np.mean(flattened_data)
    global_var = np.var(flattened_data)
    return global_mean, global_var