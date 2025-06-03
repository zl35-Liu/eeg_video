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
window_size = 100  # 每个窗口的大小，100个数据点对应0.5秒
step_size = 50     # 步长，50个数据点对应0.25秒
num_windows = 7    # 每2秒切分为7个窗口

for subname in sub_list:
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
    # 保存新数据到文件
    np.save("./data/slide7win_per2s/"+subname, new_data)import numpy as np
import os
import mne
from tqdm import tqdm

def create_epochs(data, sfreq, window_sec, step_sec=None, num_windows=None):
    """
    使用MNE创建固定长度的epochs

    参数：
    data : numpy数组 (n_channels, n_times)
    sfreq : 采样频率
    window_sec : 窗口长度（秒）
    step_sec : 窗口步长（秒），与num_windows二选一
    num_windows : 固定窗口数量，与step_sec二选一

    返回：
    epochs_data : numpy数组 (n_windows, n_channels, n_times)
    """
    # 创建虚拟info对象
    ch_names = [f'EEG{i:03d}' for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # 创建Raw对象
    raw = mne.io.RawArray(data, info)

    # 计算窗口参数
    window_samples = int(window_sec * sfreq)
    total_samples = data.shape[1]

    # 生成事件矩阵
    if num_windows:
        step_samples = (total_samples - window_samples) // (num_windows - 1)
    else:
        step_samples = int(step_sec * sfreq)

    events = mne.make_fixed_length_events(raw, duration=window_sec, overlap=window_sec - step_sec)

    # 创建Epochs对象
    epochs = mne.Epochs(raw, events, tmin=0, tmax=window_sec,
                       baseline=None, preload=True, verbose=False)

    return epochs.get_data()

def process_subject(sub_path, output_dir, sfreq=200,
                    window_sec=0.5, step_sec=None, num_windows=None):
    """
    处理单个被试数据

    参数：
    sub_path : 原始数据路径
    output_dir : 输出目录
    sfreq : 采样频率
    window_sec : 窗口长度（秒）
    step_sec : 窗口步长（秒）
    num_windows : 固定窗口数量
    """
    # 加载原始数据 (7, 40, 5, 62, 400)
    orig_data = np.load(sub_path)

    # 初始化新数据结构 (7, 40, 5, 62, n_windows, window_samples)
    n_blocks, n_concepts, n_clips, n_channels, n_times = orig_data.shape
    window_samples = int(window_sec * sfreq)

    # 自动计算实际窗口数量
    if num_windows:
        actual_num_windows = num_windows
    else:
        step_samples = int(step_sec * sfreq)
        actual_num_windows = (n_times - window_samples) // step_samples + 1

    new_shape = (n_blocks, n_concepts, n_clips,
                 n_channels, actual_num_windows, window_samples)
    new_data = np.zeros(new_shape)

    # 使用tqdm添加进度条
    with tqdm(total=np.prod(orig_data.shape[:3])) as pbar:
        for b in range(n_blocks):
            for c in range(n_concepts):
                for v in range(n_clips):
                    # 获取当前数据立方体 (62, 400)
                    cube_data = orig_data[b, c, v]

                    # 使用MNE进行epoch切分
                    epochs_data = create_epochs(
                        cube_data,
                        sfreq=sfreq,
                        window_sec=window_sec,
                        step_sec=step_sec,
                        num_windows=num_windows
                    )  # (n_windows, 62, window_samples)

                    # 重组维度
                    epochs_data = epochs_data.transpose(1, 0, 2)  # (62, n_windows, window_samples)

                    # 存入新数组
                    new_data[b, c, v] = epochs_data
                    pbar.update(1)

    # 保存处理结果
    sub_name = os.path.basename(sub_path)
    output_path = os.path.join(output_dir, sub_name)
    np.save(output_path, new_data)
    print(f"Processed {sub_name}: New shape {new_data.shape}")

# 使用示例
if __name__ == "__main__":
    # 参数配置
    config = {
        "sfreq": 200,          # 采样频率
        "window_sec": 0.5,      # 窗口长度（秒）
        "step_sec": None,       # 步长（秒）
        "num_windows": 7,       # 固定窗口数量
        "input_dir": "./data/Segmented_Rawf_200Hz_2s",
        "output_dir": "./data/slide_window_processed"
    }

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)

    # 获取被试列表
    sub_list = [f for f in os.listdir(config["input_dir"]) if f.endswith(".npy")]

    # 处理所有被试
    for sub_file in sub_list:
        sub_path = os.path.join(config["input_dir"], sub_file)
        process_subject(
            sub_path=sub_path,
            output_dir=config["output_dir"],
            **{k:v for k,v in config.items() if k in ["sfreq", "window_sec", "step_sec", "num_windows"]}
        )