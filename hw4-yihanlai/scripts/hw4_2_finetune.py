import torch
from torchvision import models

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder


class OfficeDataset(Dataset):
    def __init__(self, img_file, csv_file, le=None):
        super(OfficeDataset, self).__init__()
        self.img_file = img_file
        self.csv_file = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if(le == None):
            le = LabelEncoder()
            le.fit(self.csv_file.label)
        self.le = le
        self.csv_file.label = le.transform(self.csv_file.label)
        self.ids = list(os.listdir(img_file))
        np.save('classes.npy', le.classes_)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.img_file, self.csv_file.iloc[index, 1])
        img = self.transform(Image.open(img_path).convert('RGB')) 
        label = self.csv_file.iloc[index, 2]

        return img, label

    def __len__(self):
        return len(self.ids)



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 65),
        )

    def forward(self, x):
        x = self.linear(x)
        return x

#############################################################
# weight = 'hw4_data/pretrain_model_SL.pt'
weight = 'backbone.pth'
train_img = 'hw4_data/office/train'
train_csv = 'hw4_data/office/train.csv' 
val_img = 'hw4_data/office/val'
val_csv = 'hw4_data/office/val.csv'
#############################################################
num_epoch = 30
lr = 0.0003

train_dataset = OfficeDataset(img_file=train_img, csv_file=train_csv)
train_dataloader = DataLoader(train_dataset,
                            shuffle=True, 
                            batch_size = 32, 
                            num_workers= 24,)

val_dataset = OfficeDataset(img_file=val_img, csv_file=val_csv)
val_dataloader = DataLoader(val_dataset, 
                            shuffle=False,
                            batch_size = 32, 
                            num_workers= 24,)


exit()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = torch.load(weight)

# define model
resnet = models.resnet50(pretrained=False)
# resnet.load_state_dict(checkpoint)
resnet.load_state_dict(checkpoint["model_state_dict"])
resnet.to(device)
classifier = Classifier().to(device)
print(resnet)
print (classifier)
# resnet.eval()

# loss functions
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer_resnet = torch.optim.AdamW(resnet.parameters(), lr=lr)
optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=lr)


best_acc = 0
for epoch in range(num_epoch):
    # ---------------------------------Training-------------------------------------
    resnet.train()
    classifier.train()

    train_loss = []
    train_acc = []
    for i, (img, label) in enumerate(train_dataloader):
        img =img.to(device)
        label = label.to(device)
        
        feature = resnet(img)
        output = classifier(feature)
        loss = criterion(output, label)

        pred = torch.argmax(output, dim=1)
        acc = (pred == label).float().mean()

        optimizer_resnet.zero_grad()
        optimizer_classifier.zero_grad()
        loss.backward()
        optimizer_resnet.step()
        optimizer_classifier.step()

        train_loss.append(loss.item())
        train_acc.append(acc.item())

    current_train_loss = sum(train_loss)/len(train_loss)
    current_train_acc = np.mean(train_acc)*100
    # ---------------------------------Validation-----------------------------------
    resnet.eval()
    classifier.eval()

    val_loss = []
    val_acc = []

    for i, (img, label) in enumerate(val_dataloader):
        img =img.to(device)
        label = label.to(device)

        feature = resnet(img)
        output = classifier(feature)
        loss = criterion(output, label)
        
        pred = torch.argmax(output, dim=1)
        acc = (pred == label).float().mean()
        
        val_loss.append(loss.item())
        val_acc.append(acc.item())

    current_val_loss = sum(val_loss)/len(val_loss)
    current_val_acc = np.mean(val_acc)*100

    # print the performance
    print(f'Epoch: {epoch}/{num_epoch}, Training Loss: {current_train_loss:.4f}, Training Acc: {current_train_acc:.4f}, Validation Loss: {current_val_loss:.4f}, Validation Acc: {current_val_acc:.4f}')
        
    # update the model if the accuracy on the validation has improved
    if current_val_acc >=  best_acc:
        best_acc = current_val_acc
        torch.save({'epoch': epoch, 
                    'model_state_dict': resnet.state_dict(), 
                    'optimizer_state_dict': optimizer_resnet.state_dict(),
                    'loss': loss,}, 'model_best_resnet.pth')

        torch.save({'epoch': epoch, 
                    'model_state_dict': classifier.state_dict(), 
                    'optimizer_state_dict': optimizer_classifier.state_dict(),
                    'loss': loss,}, 'model_best_classifier.pth')
        print(f'Saving model with acc {best_acc:.4f}')

print("That's it!")