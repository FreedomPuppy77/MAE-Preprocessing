import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MAEEncoderForVA(nn.Module):
    """ MAE Encoder with a fully connected layer for VA task (Valence, Arousal) """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MAEEncoderForVA, self).__init__()
# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
        # MAE Encoder (Vision Transformer backbone)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # with cls_token
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

    def interpolate_pos_embed(self, model_state_dict):
        """
        This method interpolates position embeddings if the patch grid size of the pre-trained model
        does not match the current model's grid size (for example, when image size or patch size changes).
        """
        # Get the pos_embed from the checkpoint model state dict
        pretrain_pos_embed = model_state_dict['pos_embed']
        # Extract the original and new patch sizes
        pretrain_num_patches = pretrain_pos_embed.shape[1] - 1  # Exclude cls_token
        new_num_patches = self.pos_embed.shape[1] - 1  # Exclude cls_token

        if pretrain_num_patches != new_num_patches:
            # Only interpolate if the number of patches is different
            cls_token = pretrain_pos_embed[:, 0:1, :]  # Extract the cls_token part
            pos_embed = pretrain_pos_embed[:, 1:, :]  # Patch embedding part

            # Reshape into square grid and interpolate
            dim = int(new_num_patches**0.5)
            pos_embed = pos_embed.reshape(1, int(pretrain_num_patches**0.5), int(pretrain_num_patches**0.5), -1)
            pos_embed = F.interpolate(pos_embed, size=(dim, dim), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.reshape(1, new_num_patches, -1)

            # Concatenate cls_token back
            new_pos_embed = torch.cat([cls_token, pos_embed], dim=1)
            model_state_dict['pos_embed'] = new_pos_embed
            print(f"Position embedding interpolated from {pretrain_num_patches} to {new_num_patches} patches.")

    def load_pretrained_weights(self, weight_path):
        """
        This method loads pre-trained MAE weights, interpolates position embeddings if necessary,
        and removes decoder-related parameters from the checkpoint.
        """
        # Load the pretrained checkpoint
        checkpoint = torch.load(weight_path, map_location='cpu')
        print(f"Checkpoint keys: {checkpoint.keys()}")

        # Extract the model's state_dict from the checkpoint
        model_state_dict = checkpoint['model']

        # Interpolate position embeddings to match the model if needed
        self.interpolate_pos_embed(model_state_dict)

        # Remove decoder-related parameters from the state_dict
        state_dict = {k: v for k, v in model_state_dict.items() if 'decoder' not in k}

        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained MAE encoder weights from {weight_path}")
