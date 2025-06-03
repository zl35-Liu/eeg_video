import numpy as np
import matplotlib.pyplot as plt
import time

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics              #sklearn 评估和数据预处理
from sklearn.model_selection import train_test_split
# import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from einops import rearrange             #einops 用于张量的重排列操作（重排或变换张量形状）
import models

###################################### Hyperparameters ###################################
batch_size = 256
num_epochs = 100
lr = 0.001   # learning rate=0.001
C = 62       # the number of channels
T = 5        # the time samples of EEG signals
output_dir = './output_dir/'
network_name = "GLMNet_mlp"
# saved_model_path = output_dir + network_name + '_40c.pth'
# saved_model_path = output_dir + network_name + '_2face.pth'
# saved_model_path = output_dir + network_name + '_2human.pth'
# saved_model_path = output_dir + network_name + '_3num.pth'
saved_model_path = output_dir + network_name + '_2ofs.pth'


run_device = "cuda"

##########################################################################################

def my_normalize(data_array):
    data_array = data_array.reshape(data_array.shape[0], C*T)
    normalize = StandardScaler()  #数据按列进行标准化（即减去均值，除以标准差）
    normalize.fit(data_array)
    return (normalize.transform(data_array)).reshape(data_array.shape[0], C, T)

#a 和 b 是两个相同形状的数组，代表预测值和真实标签，tip 是输出信息的标签
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()   #ravel()展平数组
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))

#从文件名中提取出受试者的编号
def Get_subject(f):
    if(f[1] == '_'):
        return int(f[0])
    return int(f[0])*10 + int(f[1])

def Get_Dataloader(datat, labelt, istrain, batch_size):
    #datat：输入的特征数据，numpy array(num_samples, num_features)
    #labelt：输入的标签数据，(num_samples,)
    features = torch.tensor(datat, dtype=torch.float32)  #转tensor
    labels = torch.tensor(labelt, dtype=torch.long)
    return data.DataLoader(data.TensorDataset(features, labels), batch_size, shuffle=istrain)
    #TensorDataset将数据和标签封装一起，每次通过 DataLoader 迭代数据时，都会返回一个 (feature, label) 对

#累加训练中的指标（例如损失和准确率）用于统计
class Accumulator:  #@save
    def __init__(self, n):
        self.data = [0.0] * n  #初始化一个长度为 n 的列表，用于存储累加的数据

    def add(self, *args):  #接收任意数量的参数 *args，并将其与 self.data 中当前的值相加。也就是说，每次调用 add 时，都会更新 self.data 中的各个值
        self.data = [a + float(b) for a, b in zip(self.data, args)]  #将 args 中的每个元素与 self.data 中的对应元素相加

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#计时训练的各个部分，例如计算每个 epoch 所需时间
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times) / len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

#预测结果（y_hat）和真实标签（y）之间的准确度
def cal_accuracy(y_hat, y):
    #在多分类问题中，y_hat 的形状为 (batch_size, num_classes)，即每个样本都有多个类的预测分数。对于二分类问题，y_hat 通常是形状为 (batch_size, 1) 的张量，或者是形状为 (batch_size,) 的一维张量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #len(y_hat.shape) > 1，二维张量，说明输出了概率预测
        #y_hat.shape[1] > 1，第二维大于一，说明是多分类
        y_hat = torch.argmax(y_hat, axis=1)  #沿着第二维（即类别维度）选取最大值，返回每个样本的预测类别
    cmp = (y_hat == y)    #逐个比较，生成bool的tensor
    return torch.sum(cmp, dim=0)

#计算模型在验证集或测试集上的准确度
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module): #是不是pytorch网络
        net.eval()   #评估模式
        if not device:
            device = next(iter(net.parameters())).device  #设置device
    metric = Accumulator(2)  # 初始化 Accumulator 用于累计计算（累计正确数、总样本数）
    for X, y in data_iter:   #data_iter 是一个数据加载器（DataLoader）
        if isinstance(X, list):
            X = [x.to(device) for x in X]  #X 从 CPU 或其他设备迁移到指定的 device
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(cal_accuracy(net(X), y), y.numel())   #y.numel()：返回标签 y 中的样本数
        #net(X)：将输入数据 X 传递给神经网络 net，得到预测结果 y_hat。此时，net 已经在 eval() 模式下，返回的是模型的预测输出
    return metric[0] / metric[1]

def topk_accuracy(output, target, topk=(1, )):       
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    #target：真实标签，形状为 (batch_size, )
    #topk：元祖，计算哪些k下的accu
    with torch.no_grad():   #不计算梯度，推理阶段
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  #topk函数获取naxk个最大预测值和他们的索引
        pred = pred.t()   #.t() 转置索引矩阵，将其变为 (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #.t() 转置索引矩阵，将其变为 (maxk, batch_size)
        #通过 .expand_as(pred) 将 target 广播到与 pred 相同的形状 (maxk, batch_size)
        #eq(...)：返回一个布尔矩阵，表示每个预测标签是否等于真实标签
        res = []
        for k in topk:    #每个要输出的k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            #float().sum(0, keepdim=True)：将布尔值转换为浮点数（True 为 1，False 为 0），然后按列求和，得到在 Top-K 中每个样本是否正确预测的总和
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res

#训练函数
def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, save_path):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)  #使用 Xavier 均匀分布 来初始化模型中的线性层（nn.Linear）和卷积层（nn.Conv2d）的权重
    net.apply(init_weights)                    #将 init_weights 函数应用到模型 net 的所有模块中
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    #AdamW 是 Adam 优化器的变种，通过引入 权重衰减（L2 正则化）来减小过拟合的可能性。这里使用学习率 lr 并且没有使用额外的权重衰减（weight_decay=0）
    loss = nn.CrossEntropyLoss()
    
    timer, num_bathces = Timer(), len(train_iter)
    
    best_test_acc = -1
    
    best_model = net
    
    best_val_acc, best_test_acc = 0, 0   #记录最佳 模型、准确度


    #训练循环
    for epoch in range(num_epochs):
        metric = Accumulator(3)  #保存了每个 epoch 中的损失、准确度以及处理过的样本数量，供后续计算平均值
        net.train()
        for i, (X, y) in enumerate(train_iter): #dataloader每次返回一个数据和标签对
            timer.start()
            optimizer.zero_grad()         #清空之前的梯度（因为默认情况下，PyTorch 会累积梯度）
            X, y = X.to(device), y.to(device) #将输入 转移到device
            y_hat = net(X)                    #得到输出y_hat
            l = loss(y_hat, y)
            l.backward()                      #bp 计算grad
            optimizer.step()                  #upgrade weights
            metric.add(l * X.shape[0], cal_accuracy(y_hat, y), X.shape[0])    # X 应为 62*5
            timer.stop()
            train_l = metric[0] / metric[2]   #平均损失
            train_acc = metric[1] / metric[2] #平均acc
            # if(i + 1) % (num_bathces // 5) == 0 or i == num_bathces:
            #     print(f'{epoch + (i+1) / num_bathces} : , train_l:{train_l}, train_acc:{train_acc}')

        #每个ephoch后 评估和模型保存
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        val_acc = evaluate_accuracy_gpu(net, val_iter)
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save(net, save_path)
        
        if(epoch % 3 == 0):
            print(epoch)
            print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, val acc {val_acc:.3f},'f'test acc {test_acc:.3f}')
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
            # print('Saving model!')
            # # Save model to specified path
            # torch.save(net.state_dict(), './NeuralNetworkLvBaoliang/CNN_hw3/' + 'Alexnet.pth')
    # test_pred = np.array([])
    # for X, y in test_iter:
    #     X, y = X.to(device), y.to(device)
    #     y_hat = net(X)
    #     y_hat = torch.argmax(y_hat, axis=1)
    #     y_hat = y_hat.cpu().numpy()
    #     y_hat = y_hat.reshape(y_hat.shape[0])
    #     test_pred = np.concatenate((test_pred,y_hat))    
    
    # torch.save(net, save_path)
    return best_val_acc   


#定义每个类别的标签-40类
# GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
#              34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
#             [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
#  	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
#             [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
#              2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
#             [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
#             7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
#             [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
#              2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
#             [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
#              26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
#             [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
#              31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
#             ])

#标签-
# GT_label = np.load('../dataset/meta_info/All_video_face_apperance.npy')   # 初始将list转np 尝试直接np load
# GT_label = np.load('../dataset/meta_info/All_video_human_apperance.npy')   # 初始将list转np 尝试直接np load
# GT_label = np.load('../dataset/meta_info/All_video_obj_number.npy')   # 初始将list转np 尝试直接np load
GT_label = np.load('../dataset/meta_info/All_video_optical_flow_score.npy')   # 初始将list转np 尝试直接np load

GT_label = np.where(GT_label >1.799,1,0)
GT_label= GT_label.astype(np.int64)
# GT_label=np.max(reshaped_GT, axis=2)
# print(GT_label.shape)
print(GT_label)

# 直接从npy导入的标签里面是从0开始 则去掉-1
# GT_label = GT_label - 1   #要从0开始 比标签-1

#将标签从每个数据块复制 10 次，并将它们连接成一个大数组。最终的 All_label 维度为 (7, 400)，每个块的标签重复 10 次
#repeat是每个元素在自己之后重复！！！
# All_label = np.empty((0, 400))
# for block_id in range(7):
#     All_label = np.concatenate((All_label, GT_label[block_id].repeat(10).reshape(1, 400)))   #
All_label = np.empty((0, 400))
for block_id in range(7):
    All_label = np.concatenate((All_label, GT_label[block_id].repeat(2).reshape(1, 400)))

def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files_names.append(filename)
    return files_names

#获取所有数据文件
# file_path="../EEG_preprocessing/data/data/DE_1per1s/"
# file_path="../EEG_preprocessing/data/Segmented_Rawf_200Hz_2s/"
sub_list = get_files_names_in_directory("../EEG_preprocessing/data/DE_1per1s/")
# sub_list = get_files_names_in_directory(file_path)

All_sub_top1 = []
All_sub_top5 = []

for subname in sub_list:
    load_npy = np.load("../EEG_preprocessing/data/DE_1per1s/" + subname)
    # load_npy = np.load(file_path + subname)

    print(load_npy.shape)

    all_test_label = np.array([])
    all_test_pred = np.array([])

    print(load_npy.shape)

    print("shape = ", load_npy.shape)

    All_train = rearrange(load_npy, 'a b c d e f -> a (b c d) e f') #reshape   7 40 5 2 62 5
    print(All_train.shape)

    Top_1 = []
    Top_K = []

    #代码使用 7 折交叉验证方法，通过遍历不同的测试集（test_set_id）来训练和评估模型。
    #在每次迭代中，模型会使用其他的数据块作为训练集，并将一个数据块用于验证集，另一个数据块用于测试集
    for test_set_id in range(7):
        val_set_id = test_set_id - 1  #test前一个是val
        if(val_set_id < 0):
            val_set_id = 6

        #加载三部分数据
        train_data = np.empty((0, 62, 5))
        train_label = np.empty((0))
        for i in range(7):
            if(i == test_set_id):
                continue
            train_data = np.concatenate((train_data, All_train[i].reshape(400, 62, 5))) # 40*5*2s 62导 5频带 的DE
            train_label = np.concatenate((train_label, All_label[i]))
        test_data = All_train[test_set_id]
        test_label = All_label[test_set_id]
        val_data = All_train[val_set_id]
        val_label = All_label[val_set_id]

        train_data = train_data.reshape(train_data.shape[0], 62*5)
        test_data = test_data.reshape(test_data.shape[0], 62*5)
        val_data = val_data.reshape(val_data.shape[0], 62*5)

        #对训练数据和测试数据归一化
        normalize = StandardScaler()
        normalize.fit(train_data)
        train_data = normalize.transform(train_data)  #分别进行归一化
        normalize = StandardScaler()
        normalize.fit(test_data)
        test_data = normalize.transform(test_data)
        normalize = StandardScaler()
        normalize.fit(val_data)
        val_data = normalize.transform(val_data)
            
        # modelnet = models.glfnet_mlp(out_dim=40, emb_dim=64, input_dim=310)
        modelnet = models.glfnet_mlp(out_dim=2, emb_dim=64, input_dim=310)  ##emb_dim是两块网络中间输出的embeding维度
        # backdoor_net = nn.Sequential(nn.Flatten(), nn.Linear(200, 256), nn.ReLU(),
        #                              nn.Linear(256, 256), nn.ReLU(),
        #                              nn.Linear(256, 256), nn.ReLU(),
        #                              nn.Linear(256, 5))

        # norm_backdoor_train_data = my_normalize(backdoor_data).reshape(backdoor_data.shape[0], 1, C, T)
        # norm_test_data = my_normalize(test_data).reshape(test_data.shape[0], 1, C, T)

        norm_train_data = train_data.reshape(train_data.shape[0], C, T)   #变回62*5
        norm_test_data = test_data.reshape(test_data.shape[0], C, T)
        norm_val_data = val_data.reshape(val_data.shape[0], C, T)
        #三部分的数据加载器
        train_iter = Get_Dataloader(norm_train_data, train_label, istrain=True, batch_size=batch_size)
        test_iter = Get_Dataloader(norm_test_data, test_label, istrain=False, batch_size=batch_size)
        val_iter = Get_Dataloader(norm_val_data, val_label, istrain=False, batch_size=batch_size)

        now_pred = np.array([])

        accu = train(modelnet, train_iter, val_iter, test_iter, num_epochs, lr, run_device, save_path=saved_model_path)
        
        print("acc_for_train : =", accu)


        #模型评估与测试

        loaded_model = torch.load(saved_model_path,weights_only=False)   #解决报错，weights_only设置false
        loaded_model.to(run_device)   #加载训练好的模型并移到device

        block_top_1 = []
        block_top_k = []
        #每次测试集的批次数据都会输入模型，得到预测结果（y_hat）。然后计算 Top-1 和 Top-5 的准确率，并将预测结果与真实标签进行比较
        for X, y in test_iter:
            X, y = X.to(run_device), y.to(run_device)
            y_hat = loaded_model(X)

            top_K_results = topk_accuracy(y_hat, y, topk=(1,2))   # 改2
            block_top_1.append(top_K_results[0])
            block_top_k.append(top_K_results[1])

            y_hat = torch.argmax(y_hat, axis=1)
            y_hat = y_hat.cpu().numpy()
            y_hat = y_hat.reshape(y_hat.shape[0])
            all_test_pred = np.concatenate((all_test_pred,y_hat))    
        all_test_label = np.concatenate((all_test_label, test_label))

        top_1_acc = np.mean(np.array(block_top_1))
        top_k_acc = np.mean(np.array(block_top_k))
        print("top1_acc = ", top_1_acc)
        print("top5_acc = ", top_k_acc)
        Top_1.append(top_1_acc)
        Top_K.append(top_k_acc)
        
        # break

    print(metrics.classification_report(all_test_label, all_test_pred))  #生成报告
    np.seterr(divide='ignore', invalid='ignore')

    #计算 混淆矩阵，统计每个类别的正确分类数和总数
    # proper = [0] * 40
    # num = [0] * 40
    # conf = np.zeros(shape=(40, 40))
    # for i in range(all_test_label.shape[0]):
    #     l = int(all_test_label[i])
    #     p = int(all_test_pred[i])
    #     num[l] = num[l] + 1
    #     if(l == p):
    #         proper[l] = proper[l] + 1
    #     conf[l][p] = conf[l][p] + 1

    # print('confusion_matrix=eev======================================')
    # print(conf)
    # print(proper)
    # print(num)
    #
    # print("test_accu = ", np.sum(np.array(proper)) / np.sum(np.array(num)))

    print("test_Top_1_accu = ", np.mean(np.array(Top_1)))
    print("test_Top_5_accu = ", np.mean(np.array(Top_K)))
    All_sub_top1.append(np.mean(np.array(Top_1)))
    All_sub_top5.append(np.mean(np.array(Top_K)))

    save_results = np.concatenate((all_test_pred.reshape(1, all_test_label.shape[0]), all_test_label.reshape(1, all_test_label.shape[0])))
    print(save_results.shape)

    # save_dir = './ClassificationResults/2face_top1/'
    # save_dir = './ClassificationResults/2human_top1/'
    # save_dir = './ClassificationResults/3num_top1/'
    save_dir = './ClassificationResults/2ofs_top1/'
    os.makedirs(save_dir, exist_ok=True)
    # np.save('./ClassificationResults/40c_top1/'+network_name+'_Predict_Label_' + subname + '.npy', save_results)
    np.save(save_dir+network_name+'_Predict_Label_' + subname + '.npy', save_results)

    # break


#在所有交叉验证完成后，输出所有子任务的Top-1和Top-5准确率的平均值和标准差
print(All_sub_top1)
print(All_sub_top5)

print("TOP1: ", np.mean(np.array(All_sub_top1)), np.std(np.array(All_sub_top1)))
print("TOP5: ", np.mean(np.array(All_sub_top5)), np.std(np.array(All_sub_top5)))

# np.save('./ClassificationResults/40c_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))
# np.save('./ClassificationResults/40c_top5/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top5))

# np.save('./ClassificationResults/2face_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))

# np.save('./ClassificationResults/2human_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))

# np.save('./ClassificationResults/3num_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))

np.save('./ClassificationResults/2ofs_top1/'+network_name+'_All_subject_acc.npy', np.array(All_sub_top1))
