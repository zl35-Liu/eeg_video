import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 假设数据已加载为numpy数组
# 假设 de_features 形状为 (200, 62, 5), labels 形状为 (200,)
# 请替换为实际数据加载代码
de_features = np.load("/home/bcilab/Ljx/EEG2Video-main/EEG_preprocessing/data/DE_1per1s/sub1.npy.npy")
print(de_features.shape)
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

# # 模拟数据示例（仅供测试）
# np.random.seed(42)
# de_features = np.random.randn(200, 62, 5)  # 随机数据代替真实DE特征
# labels = np.random.randint(0, 40, size=200)  # 随机生成40类标签

# ---------------------------------
# 步骤1: 数据预处理
# ---------------------------------
# 展平特征: [200, 62, 5] -> [200, 62*5 = 310]
n_samples, n_channels, n_bands = de_features.shape
X_flatten = de_features.reshape(n_samples, -1)

# 标准化特征（推荐）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flatten)

# ---------------------------------
# 步骤2: t-SNE降维
# ---------------------------------
# 初始化t-SNE模型（调整参数可优化可视化效果）
tsne = TSNE(
    n_components=2,  # 降维到2D
    perplexity=30,   # 典型值在5-50之间，对高维数据适当增大
    learning_rate=200,
    random_state=42
)

# 执行降维
X_tsne = tsne.fit_transform(X_scaled)

# ---------------------------------
# 步骤3: 可视化
# ---------------------------------
plt.figure(figsize=(12, 8))

# 创建颜色映射（40类需要足够多的颜色）
cmap = plt.cm.get_cmap('tab20', 40)  # 使用'tab20'调色板，支持40种颜色

# 绘制散点图
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=labels,
    cmap=cmap,
    alpha=0.7,
    edgecolors='w',
    s=40  # 点大小
)

# 添加颜色条和图例
cbar = plt.colorbar(scatter, ticks=range(40))
cbar.set_label('Class ID')
cbar.set_ticks(np.arange(40) + 0.5)
cbar.set_ticklabels(np.arange(40))

# 标题和坐标轴
plt.title('t-SNE Visualization of DE Features (40 Classes)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 优化布局
plt.tight_layout()
plt.show()