import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
import matplotlib
matplotlib.use('Agg')


# Define constants for ImageNet mean and std
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title='', save_path=None):
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    # plt.title(title, fontsize=16)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the image to the provided path
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, save_dir):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    plt.rcParams['figure.figsize'] = [24, 24]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    show_image(x[0], save_path=os.path.join(save_dir, 'original.png'))
    show_image(im_masked[0], save_path=os.path.join(save_dir, 'masked.png'))
    show_image(y[0], save_path=os.path.join(save_dir, 'reconstruction.png'))
    show_image(im_paste[0], save_path=os.path.join(save_dir, 'reconstruction_visible.png'))

    # plt.subplot(1, 4, 1)
    # show_image(x[0], "original", save_path=os.path.join(save_dir, 'original.png'))

    # plt.subplot(1, 4, 2)
    # show_image(im_masked[0], "masked", save_path=os.path.join(save_dir, 'masked.png'))

    # plt.subplot(1, 4, 3)
    # show_image(y[0], "reconstruction", save_path=os.path.join(save_dir, 'reconstruction.png'))

    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible", save_path=os.path.join(save_dir, 'reconstruction_visible.png'))

    print(f"Image saved in {save_dir}")

def main():

    # img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
    # img = Image.open(requests.get(img_url, stream=True).raw)
    img_path = '/data/lyh/mae_demo/image/00032.jpg'
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    img = img - imagenet_mean
    img = img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(torch.tensor(img))

    # Download checkpoint if not exist
    # os.system('wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth')

    # chkpt_dir = '/data/lyh/AffectNet/AffectNet_log/checkpoint-160.pth'
    chkpt_dir = '/data/lyh/mae_visualize_vit_large.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    # Make random mask reproducible
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    save_dir = '/data/lyh/mae_demo/image_out'
    run_one_image(img, model_mae, save_dir)

if __name__ == "__main__":
    main()
