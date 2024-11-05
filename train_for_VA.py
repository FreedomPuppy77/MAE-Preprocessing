import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import random
from PIL import Image
import numpy as np
import logging

from model_for_VA import mae_vit_large_for_va  # 假设模型定义在model.py中

# 创建日志文件夹
if not os.path.exists("logs"):
    os.makedirs("logs")

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/training_log.log', filemode='w')
logger = logging.getLogger()

# 自定义数据集类
class VADataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data = []

        # 加载每个子文件夹中的图片和标签
        for subdir in os.listdir(label_dir):
            # label_files = [f"{subdir}.txt", f"{subdir}.mp4.txt", f"{subdir}.mp4_left.txt", f"{subdir}.mp4_right.txt"]
            # label_file_path = None
            # for label_file in label_files:
            #     label_file_path = os.path.join(label_path, label_file)
            #     if os.path.exists(label_file_path):
            #         break
            # if not label_file_path or not os.path.exists(label_file_path):
            #     logger.error(f"Label file not found for subdirectory {subdir}")
            #     continue
            print(f"Loading subdirectory: {subdir}")
            label_path = os.path.join(label_dir, subdir)
            img_path = os.path.join(img_dir, subdir)
            if os.path.isdir(label_path) and os.path.isdir(img_path):
                label_file = f"{subdir}.txt"
                if not os.path.exists(os.path.join(label_path, label_file)):
                    label_file = f"{subdir}.mp4.txt"
                    if not os.path.exists(os.path.join(label_path, label_file)):
                        label_file = f"{subdir}.mp4_left.txt"
                        if not os.path.exists(os.path.join(label_path, label_file)):
                            label_file = f"{subdir}.mp4_right.txt"
                label_file_path = os.path.join(label_path, label_file)
                
                try:
                    # 读取标签
                    valences, arousals = [], []
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines[1:]:  # 跳过第一行（假设是表头）
                            valence, arousal = line.strip().split(",")
                            valences.append(float(valence))
                            arousals.append(float(arousal))
                    
                    # 记录原始标签数和图片数
                    original_image_count = len(os.listdir(img_path))
                    original_label_count = len(valences)

                    # 获取图像数量
                    image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])
                    valid_data = []
                    
                    # 确保每个图片的标签正确对应
                    for img_name in image_files:
                        img_idx = int(img_name.split('.')[0])  # 假设图片名是00001.jpg的形式
                        if img_idx >= len(valences):
                            continue
                        val, aro = valences[img_idx], arousals[img_idx]
                        img_file = os.path.join(img_path, img_name)

                        valid_data.append((img_file, val, aro))

                    

                    # 记录处理后的标签数和图片数
                    filtered_image_count = len(valid_data)
                    filtered_label_count = len(valid_data)
                    print(f"Subdirectory: {subdir}, Original images: {original_image_count}, Original labels: {original_label_count}, Filtered images: {filtered_image_count}, Filtered labels: {filtered_label_count}")

                    # 添加有效的数据对到self.data
                    self.data.extend(valid_data)
                except Exception as e:
                    logger.error(f"Error loading folder {subdir}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, val, aro = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([val, aro], dtype=torch.float32)
        logger.debug(f"Accessing index {idx}, Label: {label}")
        return image, label

# CCC计算函数
def concordance_ccc(preds, labels):
    preds_mean = np.mean(preds)
    labels_mean = np.mean(labels)
    covariance = np.mean((preds - preds_mean) * (labels - labels_mean))
    preds_var = np.var(preds)
    labels_var = np.var(labels)
    ccc = (2 * covariance) / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc
def train_model(model, dataset, val_dataloader, criterion, optimizer, scheduler, num_epochs=25, patience=10, device="cuda"):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    total_size = len(dataset)
    subset_size = min(total_size, 500000)
    early_stopping_counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        print(f"Starting training for epoch {epoch}")
        logger.info(f"Starting training for epoch {epoch}")
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 20)

        # 动态创建新的训练子集
        indices = random.sample(range(total_size), subset_size)  # 随机采样 50 万张图片的索引
        train_subset = Subset(dataset, indices)  # 使用采样的索引创建新的数据子集
        train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)  # 创建新的 DataLoader

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_dataloader
            else:
                model.eval()  # 设置模型为评估模式
                dataloader = val_dataloader

            running_loss = 0.0
            valence_preds, valence_labels = [], []
            arousal_preds, arousal_labels = [], []

            # 迭代数据
            for step, (inputs, labels) in enumerate(dataloader):
                if step % 500 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch}, Phase: {phase}, Step: {step}/{len(dataloader)}, Lr: {current_lr}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播 + 反向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化，只有在训练阶段
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        # 监控cls token梯度 如果cls token接近0 表示没有有效学习
                        # for name, param in model.named_parameters():
                        #     if "cls_token" in name:
                        #         logger.info(f"{name} gradient: {param.grad}")
                        optimizer.step()

                # 统计损失
                running_loss += loss.item() * inputs.size(0)
                valence_preds.extend(outputs[:, 0].detach().cpu().numpy())
                valence_labels.extend(labels[:, 0].cpu().numpy())
                arousal_preds.extend(outputs[:, 1].detach().cpu().numpy())
                arousal_labels.extend(labels[:, 1].cpu().numpy())

            # 计算每个epoch的损失和其他评估指标
            epoch_loss = running_loss / len(dataloader.dataset)
            valence_ccc = concordance_ccc(np.array(valence_preds), np.array(valence_labels))
            arousal_ccc = concordance_ccc(np.array(arousal_preds), np.array(arousal_labels))
            valence_rmse = np.sqrt(np.mean((np.array(valence_labels) - np.array(valence_preds)) ** 2))
            arousal_rmse = np.sqrt(np.mean((np.array(arousal_labels) - np.array(arousal_preds)) ** 2))

            logger.info(f'{phase.capitalize()} Epoch: {epoch} Loss: {epoch_loss:.4f} Valence RMSE: {valence_rmse:.4f} '
                        f'Arousal RMSE: {arousal_rmse:.4f} Valence CCC: {valence_ccc:.4f} Arousal CCC: {arousal_ccc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)  # 使用验证集损失来调整学习率

                # 保存当前模型权重（包括最优和最新的模型）
                checkpoint_path = f'logs/checkpoint-{epoch}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f'Model checkpoint saved to {checkpoint_path}')

                # 保存最佳模型
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()  # 保存最佳模型权重
                    torch.save(best_model_wts, 'logs/best_model.pth')
                    early_stopping_counter = 0  # 重置早停计数器
                    logger.info(f'Best model updated at epoch {epoch} with loss {best_loss:.4f}')
                else:
                    early_stopping_counter += 1
                    logger.info(f'Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}')

                # 如果早停计数器达到设定的 patience，停止训练
                if early_stopping_counter >= patience:
                    logger.info('Early stopping triggered. Stopping training...')
                    model.load_state_dict(best_model_wts)  # 加载最优模型权重
                    return model

    # 训练结束后加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.BICUBIC, antialias=True),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.471, 0.363, 0.333], std=[0.220, 0.193, 0.180])  # 归一化
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.BICUBIC, antialias=True), 
            transforms.CenterCrop(224),# 从缩放后的图像中心裁剪224x224的区域
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.471, 0.363, 0.333], std=[0.220, 0.193, 0.180])
        ]),
    }

    # 数据集路径
    train_img_dir = "/data/lyh/Affwild2/cropped_aligned/fine_train"
    val_img_dir = "/data/lyh/Affwild2/cropped_aligned/fine_val"
    train_label_dir = "/data/lyh/Affwild2/6th_ABAW_Annotations/fine_train"
    val_label_dir = "/data/lyh/Affwild2/6th_ABAW_Annotations/fine_val"

    # 创建数据集和数据加载器
    train_dataset = VADataset(train_img_dir, train_label_dir, transform=data_transforms['train'])  # 【保持不变】使用完整训练数据集
    val_dataset = VADataset(val_img_dir, val_label_dir, transform=data_transforms['val'])
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    

    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 模型、损失函数和优化器
    model = mae_vit_large_for_va()
    logger.info(model)
    # 加载预训练权重
    checkpoint_path = "/data/lyh/mae_visualize_vit_large.pth"  # 请替换为实际的预训练权重路径
    model.load_pretrained_weights(checkpoint_path)
    model.freeze_layers()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4) #如果还过拟合设置更大的weight_decay=1e-3, 但是验证集上表现不好,可以减少
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    # 训练模型
    model = train_model(model, train_dataset, val_dataloader, criterion, optimizer, scheduler, num_epochs=20, patience=8, device=device)

    # 保存最佳模型
    torch.save(model.state_dict(), 'logs/best_va_model.pth')
