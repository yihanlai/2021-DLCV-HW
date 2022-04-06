import random
import os
import numpy as np
from PIL import Image
import sys

import torchvision
import torch
import torch.nn as nn

# # Set random seed
set_seed = 999
random.seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.cuda.manual_seed_all(set_seed)
torch.backends.cudnn.deterministic = True


class Generator(nn.Module):
    def __init__(self, img_size=28, latent_dim=100, n_classes=10):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def main():
    ######################################
    weight = 'G_model_best_ACGAN.pth'
    output =  sys.argv[1]
    ######################################
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    latent = 100
    fixed_noise = torch.randn(100, latent, device=device)
    print("The input shape:", fixed_noise.shape)
    final_list = []

    for i in range(10):
        print(f'Generating image class={i}...')
        label = torch.full((100,), i, dtype=torch.long, device=device)
        
        model = Generator().to(device)
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        fixed_img_output = model(fixed_noise, label)
        # final_list.append(fixed_img_output.cpu().data[:10])

        for j in range(100):
            torchvision.utils.save_image(fixed_img_output.cpu().data[j], f'{output}/{i}_{str(j).zfill(3)}.png',normalize=True)

    print("The output shape:",fixed_img_output.shape)

    # final_list = torch.stack(final_list)
    # final_list = torch.transpose(final_list,0,1).reshape(100,3,28,28)
    # torchvision.utils.save_image(final_list.cpu(), f'digit.png',nrow=10, normalize=True)


if __name__ == "__main__":
    main()