import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取标签文件（Valence 和 Arousal）
def load_labels_from_txt(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # 跳过标题行
    if data.ndim == 1:
        data = data.reshape(1, -1)  # 如果数据只有一行，转为二维
    valence = data[:, 0]  # 第 1 列是 Valence
    arousal = data[:, 1]  # 第 2 列是 Arousal
    return valence, arousal

# 如果标签只有一个，复制标签以匹配 patch 数量
def repeat_label(label, n_patches):
    return np.full((n_patches,), label)

# t-SNE 可视化并保存特征
def tsne_and_save_features(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
    plt.colorbar(scatter)
    plt.title("t-SNE of 04079")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close()

    print(f"t-SNE visualization saved at {save_path}")

# 加载特征文件
features = np.load("/data/lyh/mae_demo/npy/60-30-1920x1080/04079.npy")  # 替换为你的特征路径
n_patches = features.shape[0]  # 获取每张图片的 patch 数量（例如 197 个）

# 读取标签文件 (Valence 和 Arousal)
valence, arousal = load_labels_from_txt("/data/lyh/mae_demo/image/60-30-1920x1080/04079.txt")  # 替换为你的 .txt 标签路径

# 如果只有一个 Valence 标签，复制该标签以匹配 patch 数量
if len(arousal) == 1:
    arousal = repeat_label(arousal[0], n_patches)

# 调用 t-SNE 可视化并保存函数
save_path = "/data/lyh/mae_demo/npy/60-30-1920x1080/04079_arousal1.png"  # 替换为你希望保存的路径
tsne_and_save_features(features, arousal, save_path)
