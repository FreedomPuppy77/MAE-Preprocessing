import numpy as np

# 加载 .npy 文件
file_path = '/data/lyh/Affwild2/finetun_data/npy_finetun_data/train/1-30-1280x720/00001.npy'  # 替换为你的文件路径
data = np.load(file_path)

# 查看数据的形状和类型
print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

# 查看文件中的前几个数据
print("First 5 elements of the array:")
print(data[:5])  # 打印前5个元素

# 如果数据维度较高，可以查看具体某个维度的数据
print("First patch of data:")
print(data[0])  # 查看第一个patch的数据
