import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from models_mae import MaskedAutoencoderViT  # 确保模型定义文件路径正确

output_dir = '/data/lyh/Affwild2_examples/npy_data/train/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 1. 加载检查点
checkpoint = torch.load('/home/sherry/lyh/mae/Affwild2_examples/checkpoint-199.pth', map_location='cpu')
print(checkpoint.keys())  # 检查检查点中包含的键

# 2. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 定义一个函数来加载每个子文件夹中的图片
def load_dataset_from_subfolders(root_dir):
    datasets = {}
    for subdir, _, files in os.walk(root_dir):
        images = []
        if files:
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):  # 支持的图像格式
                    img_path = os.path.join(subdir, file)
                    images.append(img_path)
            datasets[os.path.basename(subdir)] = images  # 使用子文件夹名称作为键
    return datasets

# 4. 加载数据集
image_folders = load_dataset_from_subfolders('/data/lyh/Affwild2_examples/aligned/train')
print(f"Loaded folders: {image_folders.keys()}")
for folder_name, image_paths in image_folders.items():
    print(f"{folder_name}: {len(image_paths)} images")


# 5. 创建模型实例并加载状态字典
model = MaskedAutoencoderViT(img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=torch.nn.LayerNorm)
model.load_state_dict(checkpoint['model'])  # 从检查点中加载模型状态
model.eval()

# 6. 对每个子文件夹分别提取特征并保存为.npy文件
for folder_name, image_paths in image_folders.items():
    features_list = []
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB
        img = transform(img)  # 应用预处理
        images = torch.stack([img])  # 将图像堆叠为批次

        with torch.no_grad():
            features = model.forward_encoder(images, mask_ratio=0.75)[0]  # 提取特征
            num_patches = features.shape[1]
            embed_dim = features.shape[2]
            features = features.permute(0, 2, 1).reshape(1, num_patches, embed_dim).cpu().numpy()
            features_list.append(features)

    if features_list:
        all_features = np.concatenate(features_list, axis=0)
        
        # 创建保存目录（如果不存在）
        output_dir = f'/data/lyh/Affwild2_examples/npy_data/train/{folder_name}'
        os.makedirs(output_dir, exist_ok=True)  # 确保文件夹存在
        
        # 保存为.npy文件
        np.save(f'{output_dir}/mae.npy', all_features)