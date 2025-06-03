import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置
import torch
import torch.utils.data as Data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# from ..models.train_semantic_predictor import CLIP
import torch
from einops import rearrange


# VAE 产生的视频潜变量
#  latent_batch 是一个包含 200 个 latent 变量的 batch，形状为 [200, 1, 4, 6, 36, 64]
latents = torch.load('/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/test_tensor3.pt')
latents = rearrange(latents, 'a b c d e f -> a d (b c e f)')  # 重新排列张量维度
latents = latents.float()  # 确保输入是 float32

start_token = torch.zeros(200, 1, 9216)
device = latents.device  # 假设 latents 已经在正确的设备上
# 将 start_token 移动到 latents 所在的设备
start_token = start_token.to(device)
shifted_latents = torch.cat([start_token, latents[:, :-1, :]], dim=1)
# 我的生成latent长度固定6 不需要显式结束符 模型知道生成6长度


# slide_window_eeg7path='/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/slide7win_per2s/sub1.npy'
slide_window_eeg19path='/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/slide19win_per2s/sub1.npy'
slide_window_eeg7 = np.load(slide_window_eeg19path)
slide_window_eeg7 = torch.from_numpy(slide_window_eeg7)
# print("展平前eeg的shape",slide_window_eeg7.shape) #torch.Size([7, 40, 5, 62, 7, 100])  19,40

slide_window_eeg7=slide_window_eeg7[0]  # 对应 1.mp4 的脑电
# eeg=torch.from_numpy(slide_window_eeg7)
eeg=rearrange(slide_window_eeg7, 'a b c d e -> (a b) d (c e)')      # 200 62 760

# eeg = eeg[:, :6, :]  # 取前6个序列 对齐6frames
eeg = eeg.float()  # 确保输入是 float32


slide_window_eeg7test=slide_window_eeg7[6]  # 对应 7.mp4 的脑电 用于test
eeg_test=rearrange(slide_window_eeg7, 'a b c d e -> (a b) d L(c e)')
# eeg_test = eeg_test[:, :6, :].float()  # 取前6个序列 对齐6frames



# S: decoding input 的起始符
# E: decoding output 的结束符
# P：意为padding，如果当前句子短于本batch的最长句子，那么用这个符号填补缺失的单词
sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P','S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P','S i want a coke .', 'i want a coke . E'],
]

# 词典，padding用0来表示
# 源词典，本例中即德语词典
src_vocab = {'P':0, 'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
# src_vocab_size = len(src_vocab) # 6
d_en_in = 62*100  # 62*100  62个通道 窗口内100个数据点  传入维度

# 目标词典，本例中即英语词典,相比源多了特殊符
tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}
# 反向映射词典，idx —— word，原代码那个有点不好理解
idx2word = {v:k for k,v in tgt_vocab.items()}
# tgt_vocab_size = len(tgt_vocab) # 9
d_de_in = 4*1*36*64   # 9216 视频latent输入decoder的维度

src_len = 5 # 输入序列enc_input的最长序列长度，其实就是最长的那句话的token数，是指一个batch中最长呢还是所有输入数据最长呢
tgt_len = 6 # 输出序列dec_inut/dec_output的最长序列长度



# 构建模型输入的Tensor
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
    # 返回的形状为 enc_inputs：（2,5）、dec_inputs（2,6）、dec_outputs（2,6）


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        # 我们前面的enc_inputs.shape = [2,5],所以这个返回的是2
        return self.enc_inputs.shape[0]

    # 根据idx返回的是一组 enc_input, dec_input, dec_output
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# 获取输入
# enc_inputs, dec_inputs, dec_outputs = make_data(sentence)

# 构建DataLoader
# loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)
loader = Data.DataLoader(dataset=MyDataSet(eeg, shifted_latents, latents), batch_size=2, shuffle=True)
loader_test = Data.DataLoader(dataset=MyDataSet(eeg, shifted_latents, latents), batch_size=2, shuffle=True)
