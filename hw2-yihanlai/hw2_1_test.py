import random
import os
import numpy as np
import math
from PIL import Image
import sys


import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as functional


# Set random seed
set_seed = 999
random.seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.cuda.manual_seed_all(set_seed)
torch.backends.cudnn.deterministic = True

class Generator(nn.Module):
    def __init__(self, in_dim=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, in_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_dim * 8, in_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_dim * 4, in_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=4 ,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_dim, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.decoder(x)
        return output



def main():
    ######################################
    weight = 'G_model_best.pth'
    output = sys.argv[1]
    ######################################

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Generator().to(device)
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    latent = 100
    fixed_noise = torch.randn(1000, latent, 1, 1, device=device)
    print("The input shape:",fixed_noise.shape)
    fixed_img_output = model(fixed_noise)
    print("The output shape:",fixed_img_output.shape)
    # torchvision.utils.save_image(fixed_img_output.cpu().data[:32], f'face.png', nrow=8, normalize=True)

    for i in range(1000):
        torchvision.utils.save_image(fixed_img_output.cpu().data[i], f'{output}/{str(i+1).zfill(4)}.png',normalize=True)
        
if __name__ == "__main__":
    main()