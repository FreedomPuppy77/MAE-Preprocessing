import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

# 创建日志和权重保存的文件夹
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 自定义的 WarmUpLR 调度器
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, start_lr=1e-6, last_epoch=-1):
        self.total_iters = total_iters
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.start_lr + (base_lr - self.start_lr) * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]

# 自定义数据集
class VADataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        # 收集图像路径及其对应的标签路径
        for folder in os.listdir(self.img_dir):
            img_folder = os.path.join(self.img_dir, folder)
            label_file = os.path.join(self.label_dir, folder + '.txt')

            if os.path.isdir(img_folder) and os.path.exists(label_file):
                img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])

                with open(label_file, 'r') as f:
                    labels = f.readlines()[1:]

                valid_samples = []
                for img_file in img_files:
                    frame_num = int(img_file.split('.')[0])

                    if frame_num <= len(labels):
                        valence, arousal = labels[frame_num - 1].strip().split(",")
                        valence = float(valence)
                        arousal = float(arousal)

                        if -1 <= valence <= 1 and -1 <= arousal <= 1:
                            valid_samples.append((os.path.join(img_folder, img_file), (valence, arousal)))

                self.samples.extend(valid_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (valence, arousal) = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor([valence, arousal], dtype=torch.float32)
        return image, label

# CCC计算函数
def concordance_correlation_coefficient(y_true, y_pred, eps=1e-8):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    
    cor = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    sd_true = torch.std(y_true)
    sd_pred = torch.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2
    return numerator / (denominator + eps)

# 验证函数
def evaluate_model(model, val_loader, device):
    model.eval()
    valence_rmse_total = 0.0
    arousal_rmse_total = 0.0
    valence_ccc_total = 0.0
    arousal_ccc_total = 0.0
    count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            valence_pred, arousal_pred = outputs[:, 0], outputs[:, 1]
            valence_true, arousal_true = labels[:, 0], labels[:, 1]

            valence_rmse = torch.sqrt(F.mse_loss(valence_pred, valence_true))
            arousal_rmse = torch.sqrt(F.mse_loss(arousal_pred, arousal_true))

            valence_ccc = concordance_correlation_coefficient(valence_true, valence_pred)
            arousal_ccc = concordance_correlation_coefficient(arousal_true, arousal_pred)

            valence_rmse_total += valence_rmse.item()
            arousal_rmse_total += arousal_rmse.item()
            valence_ccc_total += valence_ccc.item()
            arousal_ccc_total += arousal_ccc.item()
            count += 1

    valence_rmse_avg = valence_rmse_total / count
    arousal_rmse_avg = arousal_rmse_total / count
    valence_ccc_avg = valence_ccc_total / count
    arousal_ccc_avg = arousal_ccc_total / count

    return valence_rmse_avg, arousal_rmse_avg, valence_ccc_avg, arousal_ccc_avg

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_path, log_dir, device, warmup_iters, epoch_iters, warmup):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    warmup_scheduler = WarmUpLR(optimizer, warmup_iters, start_lr=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epoch_iters - warmup + 1, T_mult=2, eta_min=1e-8
    )

    # 确保日志和模型保存目录存在
    ensure_dir(log_dir)
    ensure_dir(os.path.dirname(weight_path))

    train_log_file = os.path.join(log_dir, "train_log.txt")
    val_log_file = os.path.join(log_dir, "val_log.txt")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_step = epoch * len(train_loader) + i
            # if current_step < warmup_iters:
            #     warmup_scheduler.step()
            # else:
            #     scheduler.step(current_step - warmup_iters)

            running_loss += loss.item()

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            if (i + 1) % 20 == 0:
                # 输出包括损失、学习率
                log_message = (f'EPOCH: {epoch+1}/{num_epochs}, Step [{i + 1}/{len(train_loader)}], '
                               f'TRAIN VALENCE RMSE: {torch.sqrt(F.mse_loss(outputs[:, 0], labels[:, 0])):.4f}, '
                               f'TRAIN AROUSAL RMSE: {torch.sqrt(F.mse_loss(outputs[:, 1], labels[:, 1])):.4f}, '
                               f'TRAIN VALENCE CCC: {concordance_correlation_coefficient(labels[:, 0], outputs[:, 0]):.4f}, '
                               f'TRAIN AROUSAL CCC: {concordance_correlation_coefficient(labels[:, 1], outputs[:, 1]):.4f}, '
                               f'Loss: {loss.item():.4f}, Learning Rate: {current_lr:.8f}\n')
                print(log_message)
                with open(train_log_file, "a") as f:
                    f.write(log_message)

        # 评估验证集
        valence_rmse_avg, arousal_rmse_avg, valence_ccc_avg, arousal_ccc_avg = evaluate_model(model, val_loader, device)
        val_loss = valence_rmse_avg + arousal_rmse_avg
        current_lr = optimizer.param_groups[0]['lr']  # 记录当前学习率
        val_log_message = (f'VALIATION:'
                           f'EPOCH: {epoch+1}/{num_epochs}, Valence RMSE: {valence_rmse_avg:.4f}, '
                           f'Arousal RMSE: {arousal_rmse_avg:.4f}, Valence CCC: {valence_ccc_avg:.4f}, '
                           f'Arousal CCC: {arousal_ccc_avg:.4f}, Learning Rate: {current_lr:.8f}\n')
        print(val_log_message)
        with open(val_log_file, "a") as f:
            f.write(val_log_message)

        # 保存模型
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(log_dir, f"checkpoint-{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型在 epoch {epoch+1} 保存，验证集损失: {val_loss:.4f}")

# 参数设置
def main():
    train_img_dir = '/data/lyh/Affwild2_examples/aligned/train'
    val_img_dir = '/data/lyh/Affwild2_examples/aligned/val'
    label_dir = '/data/lyh/Affwild2_examples/Annotations'
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-3
    weight_path = '/home/sherry/lyh/mae/checkpoints/best_model.pth'
    log_dir = '/home/sherry/lyh/mae/logs'  # 日志文件夹
    img_size = 224
    warmup = 5  # 预热的 epoch 数
    iter_per_epoch = 500  # 每个 epoch 的迭代次数
    warmup_iters = iter_per_epoch * warmup  # 预热的总步数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VADataset(img_dir=train_img_dir, label_dir=label_dir, transform=transform)
    val_dataset = VADataset(img_dir=val_img_dir, label_dir=label_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    from model_VA import MAEEncoderForVA
    model = MAEEncoderForVA()
    pretrained_weights = '/data/lyh/mae_visualize_vit_large.pth'  
    model.load_pretrained_weights(pretrained_weights)

    train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_path, log_dir, device, warmup_iters, iter_per_epoch, warmup)


if __name__ == "__main__":
    main()
