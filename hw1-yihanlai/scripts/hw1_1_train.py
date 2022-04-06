import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from PIL import Image
import math

import matplotlib.pyplot as plt

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.transforms = transforms 
        self.ids = list(os.listdir(root))

    def __getitem__(self, index):
        imgs = os.listdir(self.root)
        img = Image.open(os.path.join(self.root, imgs[index]))
        label = int(imgs[index].split('_')[0])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.ids)

def get_pretrained_model(num_classes):
    # load the model pre-trained on COCO
    model = torchvision.models.vgg16_bn(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, num_classes)
    
    return model

def train_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(size=(224, 224)))
    custom_transforms.append(torchvision.transforms.RandomRotation(degrees=25))
    custom_transforms.append(torchvision.transforms.RandomPerspective(distortion_scale=0.3, p=0.5))
    custom_transforms.append(torchvision.transforms.ColorJitter(contrast=(1, 1.5)))
    custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.8))
    custom_transforms.append(torchvision.transforms.ToTensor())
    
    return torchvision.transforms.Compose(custom_transforms)

def valid_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(size=(224, 224)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    
    return torchvision.transforms.Compose(custom_transforms)


def plot_loss(train_loss, val_loss, epoch, backbone):
    num = range(epoch)
    
    plt.plot(num,train_loss, label='training loss')
    plt.plot(num, val_loss, label='validation loss')
    plt.legend()
    plt.title(f'loss (backbone={backbone})')
    plt.savefig('loss.png')

def main():
    #############################################################
    train_root = 'p1_data/train_50'
    val_root = 'p1_data/val_50'
    #############################################################

    train_dataset = MyDataset(root=train_root, transforms=train_transform())
    val_dataset = MyDataset(root=val_root, transforms=valid_transform())

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=16,)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=16,)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')

    num_classes = 50
    start_epochs = 0
    num_epochs = 50
    best_acc = 0
    backbone = 'vgg16_with_bn'

    model = get_pretrained_model(num_classes)
    model.to(device)

    print(model)
    
    #use cross-entropy as loss function
    criterion = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0001)    

    train_loss_epoch = []
    valid_loss_epoch = []
    for epoch in range(start_epochs, num_epochs, 1):
        # ---------------------------------Training-------------------------------------
        model.train()
        train_acc = []
        train_loss = []
        
        for i, (imgs, labels) in enumerate(train_dataloader):
            imgs =imgs.to(device)
            labels = labels.to(device)
            
            loss_dict = model(imgs)
            losses = criterion(loss_dict, labels)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if math.isnan(losses):
                break
            # print(f'Epoch: {epoch}/{num_epochs}, {i}/{len(train_dataloader)}')
        
            # record the training loss and acc
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

            #　using torch.no_grad() to accelerate the forward process
            with torch.no_grad():
                loss_dict = model(imgs)
            losses = criterion(loss_dict, labels)
            _, pred = torch.max(loss_dict.data, 1)
            
            # record the validation loss and acc
            val_loss.append(losses.item())
            val_acc.append((pred==labels).sum().item()/len(pred))
        
        current_val_loss = sum(val_loss)/len(val_loss)
        current_val_acc = sum(val_acc)/len(val_acc)
        valid_loss_epoch.append(current_val_loss)

        # print the performance
        print(f'Epoch: {epoch}/{num_epochs}, Training Loss: {current_train_loss}, Validation Loss: {current_val_loss}, Validation Acc: {current_val_acc}')

        # update the model if the accuracy on the validation has improved
        if current_val_acc >=  best_acc:
            best_acc = current_val_acc
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses,}, f'model_best_{backbone}.pth')
            print(f'Saving model with acc {best_acc}')

    # plot the loss figure
    plot_loss(train_loss_epoch, valid_loss_epoch, num_epochs, backbone)
    print("That's it!")
    
if __name__ == "__main__":
    main()