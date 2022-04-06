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


class Generator(nn.Module):
    # def __init__(self, img_size=28, latent_dim=100, n_classes=10):
    #     super(Generator, self).__init__()
    #     self.latent_dim = latent_dim
    #     self.label_emb = nn.Embedding(n_classes, latent_dim)

    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose2d(latent_dim, img_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
    #         nn.BatchNorm2d(img_size * 8),
    #         nn.ReLU(inplace=True),

    #         nn.ConvTranspose2d(img_size * 8, img_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
    #         nn.BatchNorm2d(img_size * 4),
    #         nn.ReLU(inplace=True),

    #         nn.ConvTranspose2d(img_size * 4, img_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
    #         nn.BatchNorm2d(img_size * 2),
    #         nn.ReLU(inplace=True),

    #         nn.ConvTranspose2d(img_size * 2, img_size, kernel_size=4 ,stride=2, padding=1, bias=False),
    #         nn.BatchNorm2d(img_size),
    #         nn.ReLU(inplace=True),

    #         nn.ConvTranspose2d(img_size, 3, kernel_size=3, stride=1, padding=3, bias=False),
    #         nn.Tanh()
    #     )

    # def forward(self, noise, labels):
    #     gen_input = torch.mul(self.label_emb(labels), noise)
    #     # print(self.label_emb(labels).shape)
    #     # print(gen_input.shape)
    #     gen_input = gen_input.view(-1, self.latent_dim,1,1)
    #     img = self.decoder(gen_input)
    #     return img


    def __init__(self, img_size=28, latent_dim=100, n_classes=10):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        # print(self.label_emb(labels).shape)
        # print(gen_input)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    # def __init__(self, img_size=28, n_classes=10):
    #     super(Discriminator, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 4 * 4, 128)
    #     self.fc2 = nn.Linear(128, 64)

    #     self.aux_fc = nn.Sequential(nn.Linear(64, 10), nn.Softmax()) 
    #     self.dis_fc = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     aux = self.aux_fc(x)
    #     dis = self.dis_fc(x)
    #     # print(aux.shape)
    #     # print(dis.shape)
    #     return dis, aux

    def __init__(self, img_size=28, n_classes=10):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(32, 0.8),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(64, 0.8),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128, 0.8),
        )

        ds_size = img_size // 14
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        # print(validity.shape)
        # print(label.shape)
        return validity, label


def get_transform(train=True):
    custom_transforms = []
    if train:
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    else:
        custom_transforms.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(custom_transforms)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def plot_loss_curve(G_loss, D_loss, epoch):
    num = range(epoch)
    
    plt.plot(num,G_loss, label='G loss')
    plt.plot(num, D_loss, label='D loss')
    plt.legend()
    plt.title(f'loss curve')
    plt.savefig('loss.png')


def main():
    ###################################################
    train_img = 'hw2_data/digits/mnistm/train'
    train_csv = 'hw2_data/digits/mnistm/train.csv'
    ###################################################

    # hyperparameters
    batch_size = 32
    latent = 100
    num_epoch = 100
    lr = 0.0002

    train_dataset = MyDataset(img_file=train_img, csv_file=train_csv, transforms=get_transform(True))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=24,)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    G = Generator().to(device)
    G.apply(weights_init)
    D = Discriminator().to(device)
    D.apply(weights_init)
    print(G)
    print(D)

    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(10, latent, device=device)
    fixed_label = torch.randint(0, 10, (batch_size, ), device=device)

    real_target = 1.
    fake_target = 0.

    G_loss_epoch = []
    D_loss_epoch = []
    best_acc = 0
    for epoch in range(0, num_epoch, 1):
        D_loss_list = []
        G_loss_list = []
        D_real_acc = 0
        D_fake_acc = 0
        G_acc = 0

        for i, (input_img, input_class) in enumerate(train_dataloader):
            # real img and label
            real_img = input_img.to(device)
            real_class = input_class.to(device)
            real_label = torch.full((real_img.size(0),1), real_target, dtype=torch.float, device=device)
            # print(f'real_img shape: {real_img.shape}')
            # print(f'real_class shape: {real_class.shape}')
            # print(f'real_label shape: {real_label.shape}')
            
            # fake img and label
            noise = torch.randn(batch_size, latent, device=device)
            # print(f'noise shape: {noise.shape}, iter: {i}')
            
            fake_class = torch.randint(0, 10, (batch_size, ),device=device)
            fake_label = torch.full((batch_size, 1), fake_target, dtype=torch.float, device=device)
            fake_image = G(noise, fake_class)
            # print(f'fake_img shape: {fake_image.shape}')
            # print(f'fake_class shape: {fake_class.shape}')
            # print(f'fake_label shape: {fake_label.shape}')
            # exit()
            #----------------------------------------Training D----------------------------------------
            D.zero_grad()

            # train with real img -> ground truth = real label
            dis_output, aux_output = D(real_img)
            D_real_dis_loss = dis_criterion(dis_output, real_label)
            D_real_aux_loss = aux_criterion(aux_output, real_class)
        
            D_real_loss = (D_real_dis_loss + D_real_aux_loss)
            # classifier acc
            pred = torch.max(aux_output, 1)[1]
            D_real_acc+=((pred==real_class).sum().item() / batch_size)
            #　generator acc
            # D_real_acc += np.mean(((dis_output > 0.5).cpu().data.numpy() == real_label.cpu().data.numpy()))
            # D_real_loss.backward()

            # train with fake img -> ground truth = fake label
            dis_output, aux_output = D(fake_image.detach())
            D_fake_dis_loss = dis_criterion(dis_output, fake_label)
            D_fake_aux_loss = aux_criterion(aux_output, fake_class)
            
            D_fake_loss = (D_fake_dis_loss + D_fake_aux_loss)

            # classifier acc
            pred = torch.max(aux_output, 1)[1]
            D_fake_acc+=((pred==fake_class).sum().item() / batch_size)
            #　generator acc
            # D_fake_acc += np.mean(((dis_output > 0.5).cpu().data.numpy() == fake_label.cpu().data.numpy()))
            # D_fake_loss.backward()
            
            # update D
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            optimizerD.step()

            #----------------------------------------Training G----------------------------------------
            G.zero_grad()

            fake_label_for_G = torch.full((batch_size, 1), real_target, dtype=torch.float, device=device)
            dis_output, aux_output = D(fake_image)
            G_dis_loss = dis_criterion(dis_output, fake_label_for_G)
            G_aux_loss = aux_criterion(aux_output, fake_class)
            # classifier acc
            pred = torch.max(aux_output, 1)[1]
            G_acc+=((pred==fake_class).sum().item() / batch_size)
            
            G_loss = G_dis_loss + G_aux_loss
            G_loss.backward()
            optimizerG.step()

            D_loss_list.append(D_loss.item())
            G_loss_list.append(G_loss.item())

        current_D_loss = sum(D_loss_list)/len(D_loss_list)
        current_G_loss = sum(G_loss_list)/len(G_loss_list)

        D_loss_epoch.append(current_D_loss)
        G_loss_epoch.append(current_G_loss)

        # print the proformace
        print(f'Epoch: [{epoch+1}/{num_epoch}],\
                D Loss: {current_D_loss:.4f}, G Loss:{current_G_loss:.4f},\
                Real acc: {(D_real_acc/len(train_dataloader)):.4f}, Fake acc: {(D_fake_acc/len(train_dataloader)):.4f}, G acc: {G_acc/len(train_dataloader):.4f}')
        
        current_val_acc = D_real_acc/len(train_dataloader)
        
        # update the model if the accuracy on the validation has improved
        if current_val_acc >=  best_acc:
            best_acc = current_val_acc
            torch.save({'epoch': epoch, 
                        'model_state_dict': G.state_dict(), 
                        'optimizer_state_dict': optimizerG.state_dict(),
                        'loss': G_loss,}, 'G_model_best_ACGAN.pth')
            torch.save({'epoch': epoch, 
                        'model_state_dict': D.state_dict(), 
                        'optimizer_state_dict': optimizerD.state_dict(),
                        'loss': G_loss,}, 'D_model_best_ACGAN.pth')
            print(f'Saving model with acc {best_acc:.4f}')

        # save sample
        # fixed_img_output = G(fixed_noise, fixed_label)
        # torchvision.utils.save_image(fixed_img_output.cpu().data, f'output/fake_epoch{epoch+1}.jpg', nrow=8, normalize=True)
        final_list = []
        for i in range(10):
            label = torch.full((10,), i, dtype=torch.long, device=device)
            G.eval()

            fixed_img_output = G(fixed_noise, label)
            final_list.append(fixed_img_output.cpu().data)
        final_list = torch.stack(final_list)
        final_list = torch.transpose(final_list,0,1).reshape(100,3,28,28)
        torchvision.utils.save_image(final_list.cpu(), f'output/fake_epoch{epoch+1}.jpg',nrow=10, normalize=True)
            
    torch.save({'epoch': epoch, 
                        'model_state_dict': G.state_dict(), 
                        'optimizer_state_dict': optimizerG.state_dict(),
                        'loss': G_loss,}, 'G_model_final_ACGAN.pth')

    plot_loss_curve(D_loss_epoch, G_loss_epoch, num_epoch)

if __name__ == "__main__":
    main()
