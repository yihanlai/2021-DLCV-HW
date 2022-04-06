import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils

import timm


class CatsDogsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms 
        self.ids = list(os.listdir(root))

    def __getitem__(self, index):
        imgs = os.listdir(self.root)
        img = Image.open(os.path.join(self.root, imgs[index])).convert('RGB')
        label = int(imgs[index].split('_')[0])
        

        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.ids)



def get_transform(train=True):
    custom_transforms = []
    if train:
        custom_transforms.append(torchvision.transforms.Resize((384, 384)))     
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    else:
        custom_transforms.append(torchvision.transforms.Resize((384, 384)))
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return torchvision.transforms.Compose(custom_transforms)


def main():
    ###########################################################
    train_root = 'hw3_data/p1_data/train'
    val_root = 'hw3_data/p1_data/val'
    ###########################################################
    
    # hyperparameters
    batch_size = 16
    num_epoch = 20
    lr = 0.00001

    # load data
    train_dataset = CatsDogsDataset(root=train_root, transforms=get_transform(True))
    val_dataset = CatsDogsDataset(root=val_root, transforms=get_transform(False))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=24,)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=24,)

    # define model
    # print("Available Vision Transformer Models: ")
    # model_names = timm.list_models("vit*")
    # print(model_names)
    # exit()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=37)
    model.to(device)
    print(model)

    # losee function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

    train_loss_epoch = []
    valid_loss_epoch = []
    best_acc = 0

    for epoch in range(num_epoch):
        # ---------------------------------Training-------------------------------------
        model.train()
        train_acc = []
        train_loss = []
        
        for i, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        current_train_loss = sum(train_loss)/len(train_loss)
        train_loss_epoch.append(current_train_loss)
            
        # ---------------------------------Validation-----------------------------------
        model.eval()
        val_acc = []
        val_loss = []
        
        for i, (img, label) in enumerate(val_dataloader):
            img = img.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, label)
                acc = (output.argmax(dim=1) == label).float().mean()
            
            val_loss.append(loss.item())
            val_acc.append(acc)
        
        current_val_loss = sum(val_loss)/len(val_loss)
        current_val_acc = sum(val_acc)/len(val_acc)
        valid_loss_epoch.append(current_val_loss)

        # print the performance
        print(f'Epoch: {epoch}/{num_epoch}, Training Loss: {current_train_loss:.4f}, Validation Loss: {current_val_loss:.4f}, Validation Acc: {current_val_acc:.4f}')
        
        # update the model if the accuracy on the validation has improved
        if current_val_acc >=  best_acc:
            best_acc = current_val_acc
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,}, 'model_best.pth')
            print(f'Saving model with acc {best_acc:.4f}')
    
    print("That's it!")

if __name__ == "__main__":
    main()