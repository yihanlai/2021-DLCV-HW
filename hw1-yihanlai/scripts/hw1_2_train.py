import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt
import math

from model import VGG16_FCN32, VGG16_FCN8, UNet

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(f for f in os.listdir(root) if f.endswith('sat.jpg')))
        self.masks = list(sorted(f for f in os.listdir(root) if f.endswith('mask.png')))
        self.new_masks = np.empty((512,512))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        mask_path = os.path.join(self.root, self.masks[index])

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        # To tensor
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        masks = self.new_masks
        masks = torch.from_numpy(masks).long()
        
        # mask color to class
        mask = (mask >= 128)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        # print(np.unique(mask))

        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown 
        # print(np.unique(masks))
        
        return img, masks

    def __len__(self):
        return len(self.imgs)


def get_transform(train=True):
    custom_transforms = []
    if train:
        # custom_transforms.append(torchvision.transforms.Resize(256))
        custom_transforms.append(torchvision.transforms.RandomRotation(degrees=25))
        custom_transforms.append(torchvision.transforms.RandomPerspective(distortion_scale=0.3, p=0.5))
        custom_transforms.append(torchvision.transforms.ColorJitter(contrast=(1, 1.5)))
        custom_transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.8))
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        # custom_transforms.append(torchvision.transforms.Resize(256))
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return torchvision.transforms.Compose(custom_transforms)


def mIou(pred_mask, mask, num_classes):
    pred_mask = F.softmax(pred_mask, dim=1) 
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_mask = pred_mask.cpu().numpy()[0]
    mask = mask.cpu().numpy()[0]
    # pred_mask = pred_mask.contiguous().view(-1)
    # mask = mask.contiguous().view(-1)

    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred_mask == i)
        tp_fn = np.sum(mask == i)
        tp = np.sum((pred_mask == i) * (mask == i))
        if ((tp_fp + tp_fn)==0):
            mean_iou += 1/6
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6

    return mean_iou


def plot_loss_curve(train_loss, val_loss, epoch):
    num = range(epoch)
    
    plt.plot(num,train_loss, label='training loss')
    plt.plot(num, val_loss, label='validation loss')
    plt.legend()
    plt.title(f'loss curve')
    plt.savefig('loss.png')


def main():
    ###########################################################
    train_root = 'p2_data/train'
    val_root = 'p2_data/validation'
    ###########################################################

    train_dataset = MyDataset(root=train_root, transforms=get_transform(True))
    val_dataset = MyDataset(root=val_root, transforms=get_transform(False))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=16,)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=8,
                                          shuffle=False,
                                          num_workers=16,)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')
    print(f'img shape: {train_dataset[0][0].shape}, label shape: {train_dataset[0][1].shape}')

    start_epochs = 0
    num_epochs = 100
    best_acc = 0

    model = VGG16_FCN8(7)
    model.to(device)
    print(model)

    # loss function
    critertion = nn.CrossEntropyLoss()
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

    train_loss_epoch = []
    valid_loss_epoch = []
    for epoch in range(start_epochs, num_epochs, 1):
        #----------------------------------Training----------------------------------
        model.train()
        train_acc = []
        train_loss = []

        for i, (img, mask) in enumerate(train_dataloader):
            img = img.to(device)
            mask = mask.to(device)

            output = model(img)
            loss = critertion(output, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if math.isnan(loss):
                break
            
            train_loss.append(loss.item())

        current_train_loss = sum(train_loss)/len(train_loss)
        train_loss_epoch.append(current_train_loss)

        # ---------------------------------Validation---------------------------------
        model.eval()
        val_miou = []
        val_loss = []

        for i, (img, mask) in enumerate(val_dataloader):
            img = img.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                output = model(img)
            loss = critertion(output, mask)

            miou_score = mIou(output, mask, 7)
            # record the validation loss and acc
            val_loss.append(loss.item())
            val_miou.append(miou_score)

        current_val_loss = sum(val_loss)/len(val_loss)
        current_val_miou = sum(val_miou)/len(val_miou)
        valid_loss_epoch.append(current_val_loss)

        # print the performance
        print(f'Epoch: {epoch}/{num_epochs}, Training Loss: {current_train_loss:.4f}, Validation Loss: {current_val_loss:.4f}, Validation mIOU: {current_val_miou:.4f}')
        
         # update the model if the accuracy on the validation has improved
        if current_val_miou >=  best_acc:
            best_acc = current_val_miou
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,}, 'model_best.pth')
            print(f'Saving model with acc {best_acc}')


    torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,}, 'model_final.pth')

    plot_loss_curve(train_loss_epoch, valid_loss_epoch, num_epochs)


if __name__ == "__main__":
    main()
