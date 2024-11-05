import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from functools import partial
from torch.utils.data import DataLoader
from PIL import Image
from models_mae import MaskedAutoencoderViT
from concurrent.futures import ProcessPoolExecutor, as_completed # 导入多进程库
from tqdm import tqdm


output_dir = '/data/lyh/mae_demo/npy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 加载检查点
checkpoint = torch.load('/data/lyh/mae_visualize_vit_large.pth', map_location='cpu')
print(checkpoint.keys())

# 2. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 加载数据集的函数
def load_dataset_from_subfolders(root_dir):
    datasets = {}
    for subdir, _, files in os.walk(root_dir):
        images = []
        if files:
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(subdir, file)
                    images.append(img_path)
            datasets[os.path.basename(subdir)] = images
    return datasets

# 4. 加载数据集
image_folders = load_dataset_from_subfolders('/data/lyh/Affwild2/cropped_aligned/train')
print(f"Loaded folders: {image_folders.keys()}")

# 5. 创建模型实例并加载状态字典
model = MaskedAutoencoderViT(img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, 
                             decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
model.load_state_dict(checkpoint['model'])
model.eval()
# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
# 移动模型到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 6. 定义特征提取函数
def extract_features(image_paths):
    features_list = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).to(device)  # 将图像移到GPU
            img = img.unsqueeze(0)  # 增加一个维度以创建批次

            with torch.no_grad():
                features = model.forward_encoder(img, mask_ratio=0)[0]
                num_patches = features.shape[1]
                embed_dim = features.shape[2]
                features = features.permute(0, 2, 1).reshape(1, num_patches, embed_dim).cpu().numpy()
            
            features_list.append((img_path, features))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return features_list

# 7. 对每个文件夹中的图像进行批处理
for folder_name, image_paths in image_folders.items():
    # 将图像路径分成多个批次
    batch_size = 64  # 可以根据GPU的显存调整
    # 使用多进程提取特征
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Processing {folder_name}", unit="batch"):
            batch_paths = image_paths[i:i + batch_size]
            futures.append(executor.submit(extract_features, batch_paths))
        for future in tqdm(as_completed(futures), desc="Saving features", total=len(futures)):
            features_list = future.result()
            # 保存特征
            for img_path, features in features_list:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                folder_output_dir = f'{output_dir}/{folder_name}'
                os.makedirs(folder_output_dir, exist_ok=True)
                np.save(f'{folder_output_dir}/{img_name}.npy', features)
print("特征提取任务完成")
