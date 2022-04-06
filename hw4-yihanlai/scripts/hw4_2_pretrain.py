import torch
from byol_pytorch import BYOL
from torchvision import models

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm 

class ImageDataset(Dataset):
    def __init__(self, img_file):
        super(ImageDataset, self).__init__()
        self.img_file = img_file
        self.ids = list(os.listdir(img_file))
        
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        imgs = os.listdir(self.img_file)
        img = Image.open(os.path.join(self.img_file, imgs[index])).convert('RGB')
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.ids)

#########################################
train_img = 'hw4_data/mini/train'
train_csv = 'hw4_data/mini/train.csv'
#########################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

resnet = models.resnet50(pretrained=False)
learner = BYOL(resnet, image_size = 128, hidden_layer = 'avgpool')
learner.to(device)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

train_dataset = ImageDataset(train_img)
train_dataloader = DataLoader(train_dataset, 
                            batch_size = 128, 
                            num_workers=24,)

for i in range(100):
    losses = []
    print(f"Processing epoch {i}...")
    for img in tqdm(train_dataloader):
        img = img.to(device)
        loss = learner(img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        losses.append(loss.item())
    avg_loss = np.mean(losses)
    print(f"Training loss: {avg_loss:.4f}")

# save your improved network
torch.save({'model_state_dict': resnet.state_dict()}, 'backbone.pth')
