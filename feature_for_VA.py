import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image
from models_mae import MaskedAutoencoderViT  # 导入 MAE 模型类

# MAE 模型特征提取器（移除解码器）
class MAEFeatureExtractor(MaskedAutoencoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 移除解码器部分，仅保留编码器
        del self.decoder_embed
        del self.mask_token
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred
        del self.decoder_pos_embed

    def forward(self, imgs):
        # 使用编码器提取视觉特征
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)  # 不使用掩码，mask_ratio=0.0
        cls_token_feature = latent[:, 0, :]  # 提取 cls token 的特征，维度为 (batch_size, embed_dim)
        return cls_token_feature

# 加载预训练的 MAE 模型并移除解码器
def load_pretrained_mae_feature_extractor(ckpt_path):
    model = MAEFeatureExtractor(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    
    # 加载预训练的权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # state_dict = checkpoint['model']
    
    # 手动过滤掉解码器的权重，只保留编码器的权重
    encoder_state_dict = {k: v for k, v in checkpoint.items() if 'decoder' not in k}

    # 获取模型当前的参数
    model_state_dict = model.state_dict()

    # 检查编码器权重的匹配情况
    filtered_state_dict = {}
    for k, v in encoder_state_dict.items():
        if k in model_state_dict:
            filtered_state_dict[k] = v  # 只加载匹配的编码器部分权重
        else:
            print(f"Skipping key {k} as it's not in the encoder")

    # 加载过滤后的编码器权重
    model.load_state_dict(filtered_state_dict, strict=True)  # 确保严格匹配
    return model

# 输入预处理
transform = transforms.Compose([
    transforms.Resize(size=(256, 256), interpolation=Image.BICUBIC, antialias=True), 
    transforms.CenterCrop(224),# 从缩放后的图像中心裁剪224x224的区域
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.471, 0.363, 0.333], std=[0.220, 0.193, 0.180])
])

# 自定义数据集类，支持递归遍历子文件夹
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        
        # 遍历所有子文件夹
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('png', 'jpg', 'jpeg')):
                    self.img_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 打开并转换为RGB图像
        if self.transform:
            image = self.transform(image)  # 应用预处理
        return image, img_path

# 批量提取特征并保存为npy文件，保留原始目录结构
def extract_and_save_features_batch(model, dataloader, save_dir, root_dir):
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 禁用梯度计算
        for imgs, img_paths in dataloader:
            imgs = imgs.cuda()  # 使用GPU
            features = model(imgs)  # 提取 cls token 特征
            
            # 将当前批次的特征和路径保存
            save_features(features.cpu().numpy(), img_paths, save_dir, root_dir)

def save_features(batch_features, batch_paths, save_dir, root_dir):
    # 打印保存的每个特征的维度
    # print(f"Saving features with shape: {batch_features.shape}")
    # 保存每张图片的特征
    for i, feature in enumerate(batch_features):
        img_path = batch_paths[i]
        # 计算相对于根目录的路径，以便保存时保留文件夹结构
        relative_path = os.path.relpath(img_path, root_dir)
        feature_save_path = os.path.join(save_dir, os.path.splitext(relative_path)[0] + ".npy")
        
        # 创建保存目录
        os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
        
        # 保存单个特征文件
        np.save(feature_save_path, feature)
        # 打印保存的当前 .npy 文件的维度
        # print(f"Feature shape for {os.path.basename(img_path)}: {feature.shape}")
        print(f"Feature saved for {os.path.basename(img_path)} at {feature_save_path}")

# 使用方法
if __name__ == "__main__":
    # 1. 加载预训练模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/home/sherry/lyh/mae/logs/checkpoint-2.pth"  # 替换为你预训练模型的路径
    model = load_pretrained_mae_feature_extractor(ckpt_path)
    model.cuda()  # 如果使用GPU

    # 2. 定义数据集
    img_dir = '/data/lyh/Affwild2/cropped_aligned/val'  # 替换为你的图像文件夹路径
    dataset = CustomImageDataset(img_dir, transform=transform)
    
    # 使用多个线程加载数据
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    # 3. 提取特征并保存
    save_dir = "/data/lyh/Affwild2/finetun_data/npy_finetun_data/val"  # 替换为你希望保存特征的文件夹路径
    extract_and_save_features_batch(model, dataloader, save_dir, img_dir)
