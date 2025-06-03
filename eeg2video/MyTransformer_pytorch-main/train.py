import os

import torch.optim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

from torch import optim

from model import *
import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR



print("eeg的shape，应（200 6 6200）",eeg.shape)

print("shifted_latent的shape，应（200 6 9216）",shifted_latents.shape)

print("latent的shape，应（200 6 9216）",latents.shape)


model = Transformer().cuda()
model.train()
# 损失函数,忽略为0的类别不对其计算loss（因为是padding无意义）
# criterion = nn.CrossEntropyLoss(ignore_index=0)

num_epochs = 400
warmup_steps = num_epochs*0.05
# 定义预热函数
def warmup_schedule(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return 1.0  # 预热结束后保持学习率不变，后续由余弦退火接管
criterion = nn.MSELoss()  # 替换 nn.CrossEntropyLoss()  回归任务用
optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.99)   # 原本1e-3
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.005,betas=(0.9, 0.98),weight_decay=0.05)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,lr_lambda=[warmup_schedule])
# # 创建学习率调度器（结合预热和余弦退火）
# scheduler = LambdaLR(
#     optimizer,
#     lr_lambda=[warmup_schedule],  # 应用预热函数
# )
# # 如果需进一步添加余弦退火：
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=5e-6
)

train_losses = []
# 训练开始
for epoch in tqdm.tqdm(range(num_epochs)):
    epoch_loss = 0.0
    for enc_inputs, dec_inputs, dec_outputs in loader:
        # print("enc_inputs",enc_inputs.shape)
        # print("dec_inputs",dec_inputs.shape)
        # print("dec_outputs",dec_outputs.shape)
        '''
        enc_inputs: [batch_size, src_len] [2,6,6200]      19 2480 
        dec_inputs: [batch_size, tgt_len] [2,6,9216]
        dec_outputs: [batch_size, tgt_len] [2,6,9216]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        outputs = model(enc_inputs, dec_inputs) # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # outputs: [batch_size * tgt_len, tgt_vocab_size], dec_outputs: [batch_size, tgt_len]
        # loss = criterion(outputs.view(-1), dec_outputs.view(-1))  # 将dec_outputs展平成一维张量
        outputs= rearrange(outputs, '(b t) v -> b t v', b=2, t=6)  # 调成和dec_outputs一样的形状
        # print("model outpt shape ",outputs.shape)
        loss = criterion(outputs, dec_outputs)
        # 更新权重
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（范数裁剪，阈值设为1.0）
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,  # 梯度最大范数阈值
            norm_type=2  # 使用L2范数
        )

        optimizer.step()
        # print("lr",optimizer.param_groups[0]['lr'])
        epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}',"lr",optimizer.param_groups[0]['lr'])
    scheduler.step()
    avg_loss = epoch_loss / len(loader)
    train_losses.append(avg_loss)

torch.save(model, f'MyTransformer_temp415.pth')
# 绘制损失曲线
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

"""
501 d1024,en4,de8,lr0.0025,400轮,输出层新projection-19，    
"""

'''
400 - d1024 dk128 head8
401 - d2048 dk128 head16 en3 de6   loss到0.3  batch2
402 - 改401 lr5e-3 加schedule985  btch8 
403 - 改402 lr1e-4 schedule预先   batch2
404 - 改403 lr0.001 schedule余弦   batch2  200轮
405 - 改404 300轮 
406 - en和de的输入linear层加到3层（a-4096-2048-1024） d1024 400轮 lr0.001 
407 - 改406 换noisy tensor1 
408 - tensor1 d2048 400轮 lr0.001 en6 de6  loss到0.66 可能存到407了
409 - 改408 500轮 lr0.002 en4 de8    loss到0.48
410 - 改409 d1024 adamWlr0.002 en4 de8 梯度裁剪 dropout 
411 - tensor3n 参数同410 en3 de6 lr0.001 400轮    loss卡0.93
412 - 改411  lr0.001 200轮 裁剪范数10 换回SGD      loss卡0.93
413 - 改412  200轮 回d2048  lr0.001
414 - 改413  300轮  lr0.002 en4 de8

415 -  300轮  lr0.005 en4 de8 d1024 loss0.65 400loss0.65    lr0.001 400loss0.53  500loss0.5    lr0.002 400loss0.48  500loss0.41  600loss0.38 600(改min5e6）loss0.38  700loss0.33  lr0.003 500loss0.41   lr0.0025 500loss0.39  700loss0.32 700改min1e5loss0.32
416 -  300轮  改415 d2048        lr005 loss0.77 400轮loss0.4  500loss0.45

417
'''