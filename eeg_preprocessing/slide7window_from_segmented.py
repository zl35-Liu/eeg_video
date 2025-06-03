import numpy as np
import os

from accelerate.test_utils.scripts.test_script import print_on
from tqdm import tqdm  #进度条库

# segment a EEG data numpy array with the shape of (7 * 62 * 520s*fre) into 2-sec EEG segments
# segment it into a new array with the shape of (7 * 40 * 5 * 62 * 2s*fre),
# meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.

fre = 200


def analyze_npy_file(file_path):
    # 加载.npy文件
    data = np.load(file_path)

    # 输出数据的维度大小
    print(f"数据的形状为: {data.shape}")

    # 输出每个维度的大小
    for idx, dim_size in enumerate(data.shape):
        print(f"维度 {idx + 1} 的大小为: {dim_size}")


#遍历指定目录及其子目录，获取所有文件的文件名，并将它们存储到 files_names 列表中
def get_files_names_in_directory(directory):
    files_names = []
    for root, _, filenames in os.walk(directory):  #os.walk()遍历指定目录及其所有子目录，并返回一个三元组 (dirpath, dirnames, filenames)
        for filename in filenames:
            files_names.append(filename)
    return files_names

sub_list = get_files_names_in_directory("./data/Segmented_Rawf_200Hz_2s/")   #sub_list 保存了目标目录中所有 .npy 文件的文件名

print(sub_list)



# 设置窗口参数
# window_size = 100  # 每个窗口的大小，100个数据点对应0.5秒
# step_size = 50     # 步长，50个数据点对应0.25秒
# num_windows = 7    # 每2秒切分为7个窗口
window_size = 40
step_size = 20
num_windows = 19

for subname in sub_list:
    if subname != "sub1.npy":
        continue
    # 原始数据形状为 (7 ,40, 5, 62,400)
    data = np.load("./data/Segmented_Rawf_200Hz_2s/"+subname)  #
    print(data.shape,"chushi(7 ,40, 5, 62,400)")

    # 初始化切分后的数据存储数组
    new_data = np.zeros((7 ,40, 5, 62, num_windows, window_size))

    # 滑动窗口切分数据
    for p in range(7):  # 对每个样本
        for i in range(40):  # 对每个样本
            for j in range(5):  # 对每个通道
                for k in range(62):  # 对每个电极
                    for w in range(num_windows):  # 对每个窗口
                        start = w * step_size  # 窗口的起始位置
                        new_data[p,i, j, k, w, :] = data[p,i, j, k, start:start + window_size]

    # 查看新数据的形状
    print(new_data.shape," 7 ,40, 5, 62, 7, 100")  # 输出应为 (7 ,40, 5, 62, 7, 100)
    # 19个 0.2s 7 40 5 62 19 40

    # 保存新数据到文件
    np.save("./data/slide19win_per2s/"+subname, new_data)