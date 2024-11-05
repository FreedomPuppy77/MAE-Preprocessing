import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block


class MAEForVATask(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, num_classes=2):
        super().__init__()

        # --------------------------------------------------------------------------
        # 编码器部分（保留预训练的MAE模型中的部分）
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # 将图像分块并嵌入为高维向量
        num_patches = self.patch_embed.num_patches  # 获取图像中块的总数量

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 类别标记，用于汇总全局信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # 位置嵌入，使用固定的sin-cos嵌入
        # self.pre_head_dropout = nn.Dropout(0.3)
        # 使用Transformer的Block模块构建编码器部分
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  # 最后的归一化层
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # 用于VA任务的线性回归层
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),  # 归一化层
            nn.Dropout(0.4),
            nn.Linear(embed_dim, num_classes)  # 全连接层，用于输出情感的Valence和Arousal
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()  # 初始化权重

    def initialize_weights(self):
        # 初始化位置嵌入、cls标记和patch嵌入
        torch.nn.init.normal_(self.cls_token, std=.02)  # 对cls标记进行正态分布初始化
        # 初始化patch嵌入的投影层，类似于nn.Linear而不是nn.Conv2d
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def load_pretrained_weights(self, checkpoint_path):
        # 加载预训练的MAE模型权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def forward(self, imgs):
        # 编码器前向传播
        x = self.patch_embed(imgs)  # 对输入图像进行分块和嵌入

        # 添加位置嵌入（不包含cls标记）
        x = x + self.pos_embed[:, 1:, :]

        # 添加cls标记
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # 将cls标记与位置嵌入相加
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # 扩展cls标记以适应批次大小
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls标记和嵌入的patch拼接在一起

        # 通过多个Transformer Block处理嵌入特征
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 最后的归一化处理
        # x = self.pre_head_dropout(x) # 添加dropout层
        # 取cls标记的输出用于VA任务
        cls_output = x[:, 0]  # 获取cls标记的输出，代表全局特征

        # 通过线性回归头进行预测
        output = self.head(cls_output)  # 输出Valence和Arousal

        return output
    def freeze_layers(self):
        for param in self.patch_embed.parameters():
            param.requires_gard = False
        # 冻结前12层Transform Blocks
        for i in range(18):
            for param in self.blocks[i].parameters():
                param.requires_gard = False


def mae_vit_large_for_va(**kwargs):
    model = MAEForVATask(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_for_va(**kwargs):
    model = MAEForVATask(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_for_va(**kwargs):
    model = MAEForVATask(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# 示例使用
if __name__ == "__main__":
    # 创建模型实例
    model = mae_vit_large_for_va(num_classes=2)  # num_classes=2 用于情感分析中的Valence和Arousal
    
    # 加载预训练权重
    checkpoint_path = "/home/sherry/lyh/mae/logs/checkpoint-2.pth"  # 请替换为实际的预训练权重路径
    model.load_pretrained_weights(checkpoint_path)
    
    # 测试输入
    imgs = torch.randn(1, 3, 224, 224)  # 随机生成一个输入图像，大小为1x3x224x224
    
    # 前向传播
    output = model(imgs)
    print(output)  # 应输出形状为[1, 2]的张量，表示Valence和Arousal的值
