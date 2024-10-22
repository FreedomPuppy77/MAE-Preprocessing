import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MAEEncoderForVA(nn.Module):
    """ MAE Encoder with a fully connected layer for VA task (Valence, Arousal) """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MAEEncoderForVA, self).__init__()

        # MAE Encoder (Vision Transformer backbone)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Fully connected layer for VA task (output valence and arousal)
        self.fc = nn.Linear(embed_dim, 2)  # output 2 values (valence, arousal)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # class token
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Only take the class token and pass it through the FC layer
        x = self.fc(x[:, 0])
        return x

    def load_pretrained_weights(self, weight_path):
        # Load the pretrained MAE weights (for encoder)
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model']

        # Remove decoder-related parameters from the state_dict
        state_dict = {k: v for k, v in state_dict.items() if 'decoder' not in k}

        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained MAE encoder weights from {weight_path}")
