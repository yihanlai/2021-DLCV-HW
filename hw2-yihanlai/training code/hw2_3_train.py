import random
import os
import numpy as np
import math 
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, img_file, csv_file, transforms=None):
        super(MyDataset, self).__init__()
        self.img_file = img_file
        self.csv_file = pd.read_csv(csv_file)
        self.transforms = transforms
        self.ids = list(os.listdir(img_file))

    def __getitem__(self, index):

        img_path = os.path.join(self.img_file, self.csv_file.iloc[index, 0])
        img = Image.open(img_path)
        label = self.csv_file.iloc[index, 1]

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label

    def __len__(self):
        return len(self.ids)


class BackboneModel(nn.Module):
    def __init__(self, backbone = 'resnet50', n_class = 10, usps=False):
        super(BackboneModel, self).__init__()
        self.backbone = backbone
        self.n_class = n_class
        if self.backbone == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=False)
            in_features = self.model.fc.in_features
            fc = torch.nn.Linear(in_features, n_class)
            self.model.fc = fc
        self.model.fc.weight.data.normal_(0, 0.005)
        self.model.fc.bias.data.fill_(0.1)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.forward(x)


def get_transform(train=True, usps=False):
    custom_transforms = []
    if train:
        custom_transforms.append(torchvision.transforms.RandomRotation(degrees=25))
        custom_transforms.append(torchvision.transforms.RandomPerspective(distortion_scale=0.3, p=0.5))
        custom_transforms.append(torchvision.transforms.ColorJitter(contrast=(1, 1.5)))
        custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.8))
        custom_transforms.append(torchvision.transforms.ToTensor())
        
    else:
        custom_transforms.append(torchvision.transforms.ToTensor())
    if usps:
        custom_transforms.append(torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    return torchvision.transforms.Compose(custom_transforms)


def main():
    ###################################################
    train_img = 'hw2_data/digits/usps/train'
    train_csv = 'hw2_data/digits/usps/train.csv'
    val_img = 'hw2_data/digits/svhn/test'
    val_csv = 'hw2_data/digits/svhn/test.csv'
    ###################################################

    # hyperparameters
    batch_size = 64
    num_epoch = 50
    lr = 0.0001
   
    # load data
    train_dataset = MyDataset(img_file=train_img, csv_file=train_csv, transforms=get_transform(True, usps=True))
    val_dataset = MyDataset(img_file=val_img, csv_file=val_csv, transforms=get_transform(False))
    # source
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=24,)
    # target
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=24,)

    print(f'training img shape: {train_dataset[0][0].shape}')
    print(f'validation img shape: {val_dataset[0][0].shape}')

    # define model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BackboneModel().to(device)
    print(model)
    
    # loss functions
    criterion = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    
    train_loss_epoch = []
    valid_loss_epoch = []
    best_acc = 0
    
    for epoch in range(num_epoch):
        # ---------------------------------Training-------------------------------------
        model.train()
        train_acc = []
        train_loss = []
        
        for i, (imgs, labels) in enumerate(train_dataloader):
            imgs =imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)
            losses = criterion(output, labels)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss.append(losses.item())
        
        current_train_loss = sum(train_loss)/len(train_loss)
        train_loss_epoch.append(current_train_loss)
            
        # ---------------------------------Validation-----------------------------------
        model.eval()
        val_acc = []
        val_loss = []
        
        for i, (img, label) in enumerate(val_dataloader):
            imgs = img.to(device)
            labels = label.to(device)

            with torch.no_grad():
                output = model(imgs)
                losses = criterion(output, labels)
                _, pred = torch.max(output.data, 1)
            
            val_loss.append(losses.item())
            val_acc.append((pred==labels).sum().item())
        
        current_val_loss = sum(val_loss)/len(val_loss)
        current_val_acc = sum(val_acc)/len(val_acc)
        valid_loss_epoch.append(current_val_loss)

        # print the performance
        print(f'Epoch: {epoch}/{num_epoch}, Training Loss: {current_train_loss}, Validation Loss: {current_val_loss}, Validation Acc: {current_val_acc}')
        
        # update the model if the accuracy on the validation has improved
        if current_val_acc >=  best_acc:
            best_acc = current_val_acc
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,}, 'model_best_resnet50.pth')
            print(f'Saving model with acc {best_acc}')
    
    print("That's it!")

if __name__ == "__main__":
    main()

