import os
from PIL import Image
import numpy as np

# 请替换为你的数据集路径
img_root = "/data/lyh/Affwild2/cropped_aligned/train"  
means = []
stds = []

# 使用 os.walk 遍历所有子文件夹及文件
for root, _, files in os.walk(img_root):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img) / 255.0  # 将像素值标准化为 [0, 1] 范围

            # 计算每张图片的均值和标准差，并添加到列表中
            means.append(np.mean(img_np, axis=(0, 1)))
            stds.append(np.std(img_np, axis=(0, 1)))

# 计算整个数据集的均值和标准差
mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)

print(f"Dataset mean: {mean}, std: {std}")
