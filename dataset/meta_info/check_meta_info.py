import numpy as np

import torch

# # 检查 PyTorch 版本
# print("PyTorch 版本:", torch.__version__)
#
# # 检查 CUDA 是否可用
# print("CUDA 是否可用:", torch.cuda.is_available())
#
# # 如果 CUDA 可用，检查 CUDA 版本
# if torch.cuda.is_available():
#     print("CUDA 版本:", torch.version.cuda)
#     print("当前 GPU 设备:", torch.cuda.get_device_name(0))
# else:
#     print("CUDA 不可用")
#
# import torch
#
# # 创建一个随机张量并移动到 GPU
# x = torch.rand(2, 3, 224, 224).cuda()
#
# # 执行卷积操作
# conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()
# y = conv(x)
#
# print(y.shape)  # 应该输出 torch.Size([2, 16, 224, 224])

# 加载 .npy 文件
color_data = np.load('./All_video_color.npy')
face_data = np.load('./All_video_face_apperance.npy')
human_data = np.load('./All_video_human_apperance.npy')
label_data = np.load('./All_video_label.npy')
obj_number_data = np.load('./All_video_obj_number.npy')
flow_score_data = np.load('./All_video_optical_flow_score.npy')

# 设置 NumPy 打印选项，禁用省略显示
np.set_printoptions(threshold=np.inf)

# 打印每个文件的数据内容（查看数据结构、类型、维度等）
print(color_data.dtype," Color Data:", color_data)
print(face_data.dtype," Face Data:", face_data)
print(human_data.dtype," Human Appearance Data:", human_data)
print(label_data.dtype," Label Data:", label_data)
print(obj_number_data.dtype," Object Number Data:", obj_number_data)
print(flow_score_data.dtype," Optical Flow Score Data:", flow_score_data)

# face_int = face_data.astype(np.int64)
# print(list(face_int))