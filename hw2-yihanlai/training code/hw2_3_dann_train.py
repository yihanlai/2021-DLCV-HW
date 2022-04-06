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
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, img_file, csv_file, transforms=None):
        super(MyDataset, self).__init__()
        self.img_file = img_file
        self.csv_file = pd.read_csv(csv_file)
        self.transforms = transforms
        self.ids = list(os.listdir(img_file))

    def __getitem__(self, index):

        img_path = os.path.join(self.img_file, self.csv_file.iloc[index, 0])
        img = Image.open(img_path).convert('RGB')
        label = self.csv_file.iloc[index, 1]

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label

    def __len__(self):
        return len(self.ids)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        # print(f'feature extractor output:　{x.shape}')
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        # print(f'label predictor input {h.shape}')
        c = self.layer(h)
        # print(f'label predictor output {c.shape}')
        return c


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        # print(f'domain classifier intput {h.shape}')
        y = self.layer(h)
        # print(f' domain classifier output {y.shape}')
        return y

def get_transform(train=True):
    custom_transforms = []
    if train:
        custom_transforms.append(torchvision.transforms.RandomRotation(degrees=25))
        custom_transforms.append(torchvision.transforms.RandomPerspective(distortion_scale=0.3, p=0.5))
        custom_transforms.append(torchvision.transforms.ColorJitter(contrast=(1, 1.5)))
        custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.8))
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
        custom_transforms.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(custom_transforms)


def main():
    ###################################################
    source_img = 'hw2_data/digits/usps/train'
    source_csv = 'hw2_data/digits/usps/train.csv'
    target_img = 'hw2_data/digits/svhn/train'
    target_csv = 'hw2_data/digits/svhn/train.csv'
    ###################################################

    # hyperparameters
    batch_size = 64
    num_epoch = 100
    lr = 0.0001
    lamb = 0.1

    # load data
    source_dataset = MyDataset(img_file=source_img, csv_file=source_csv, transforms=get_transform(True))
    target_dataset = MyDataset(img_file=target_img, csv_file=target_csv, transforms=get_transform(True))
    
    source_dataloader = torch.utils.data.DataLoader(source_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=24,)
    target_dataloader = torch.utils.data.DataLoader(target_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=24,)

    print(f'source img shape: {source_dataset[0][0].shape}')
    print(f'target img shape: {target_dataset[0][0].shape}')

    # define model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)
    
    # loss functions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer_F = torch.optim.Adam(feature_extractor.parameters())
    optimizer_C = torch.optim.Adam(label_predictor.parameters())
    optimizer_D = torch.optim.Adam(domain_classifier.parameters())

    D_loss_epoch = []
    F_loss_epoch = []
    best_acc = 0
    
    for epoch in range(num_epoch):
        D_loss = []
        F_loss = []
        C_acc = []

        for i, ((s_img, s_label),(t_img, _)) in enumerate(zip(source_dataloader, target_dataloader)):
            s_img = s_img.to(device)
            s_label = s_label.to(device)
            t_img = t_img.to(device)
            
            mix_img = torch.cat((s_img, t_img), 0)
            domain_label = torch.full((mix_img.shape[0], 1), 0, dtype=torch.float, device=device)
            domain_label[:s_img.shape[0]] = 1

            # ---------------------------------Training Domain Classifier-------------------------------------
            feature_extractor.train()
            domain_classifier.train()
            label_predictor.train()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()
            # feature_extractor.zero_grad()
            # domain_classifier.zero_grad()
            # label_predictor.zero_grad()

            feature_output = feature_extractor(mix_img)
            domain_output = domain_classifier(feature_output.detach())
            d_loss = domain_criterion(domain_output, domain_label)
            D_loss.append(d_loss)
            d_loss.backward()
            optimizer_D.step()
            # ---------------------------------Training Feature Extractor and Domain Classifier-------------------------------------
            class_output = label_predictor(feature_output[:s_img.shape[0]])
            domain_output = domain_classifier(feature_output)
            f_loss = class_criterion(class_output, s_label) - lamb * domain_criterion(domain_output, domain_label)
            F_loss.append(f_loss)
            f_loss.backward()
            optimizer_F.step()
            optimizer_C.step()

            # print(torch.argmax(class_output, 1))
            # print(s_label)
            # print(torch.sum(torch.argmax(class_output, 1) == s_label).item())
            C_acc.append(torch.sum(torch.argmax(class_output, 1) == s_label).item()/batch_size)

        current_D_loss = sum(D_loss)/len(D_loss)
        current_F_loss = sum(F_loss)/len(F_loss)
        current_C_acc = sum(C_acc)/len(C_acc)
        D_loss_epoch.append(current_D_loss)
        F_loss_epoch.append(current_F_loss)

        # print the performance
        print(f'Epoch: {epoch}/{num_epoch}, Training D Loss: {current_D_loss:.4f}, Training F Loss: {current_F_loss:.4f}, Classifier Acc: {current_C_acc:.4f}')
        
        # update the model if the accuracy on the validation has improved
        if current_C_acc >=  best_acc:
            best_acc = current_C_acc
            torch.save({'epoch': epoch, 
                        'model_state_dict': feature_extractor.state_dict(), 
                        'optimizer_state_dict': optimizer_F.state_dict(),
                        'loss': f_loss,}, 'extractor_model.pth')

            torch.save({'epoch': epoch, 
                        'model_state_dict': label_predictor.state_dict(), 
                        'optimizer_state_dict': optimizer_C.state_dict(),
                        'loss': f_loss,}, 'label_predictor.pth')

            torch.save({'epoch': epoch, 
                        'model_state_dict': domain_classifier.state_dict(), 
                        'optimizer_state_dict': optimizer_D.state_dict(),
                        'loss': d_loss,}, 'domain_classifier.pth')

            print(f'Saving model with acc {best_acc:.4f}')

    torch.save({'epoch': epoch, 
                'model_state_dict': feature_extractor.state_dict(), 
                'optimizer_state_dict': optimizer_F.state_dict(),
                'loss': f_loss,}, 'extractor_model_final.pth')

    torch.save({'epoch': epoch, 
                'model_state_dict': label_predictor.state_dict(), 
                'optimizer_state_dict': optimizer_C.state_dict(),
                'loss': f_loss,}, 'label_predictor_final.pth')

    torch.save({'epoch': epoch, 
                        'model_state_dict': domain_classifier.state_dict(), 
                        'optimizer_state_dict': optimizer_D.state_dict(),
                        'loss': d_loss,}, 'domain_classifier_final.pth')

    print("That's it!")

if __name__ == "__main__":
    main()

