import random
import os
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy import linalg


# # Set random seed
# set_seed = 999
# random.seed(set_seed)
# torch.cuda.manual_seed(set_seed)


class MyDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.ids = list(os.listdir(root))

    def __getitem__(self, index):
        imgs = os.listdir(self.root)
        img = Image.open(os.path.join(self.root, imgs[index]))
        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.ids)

"""
DCGAN
Input Shape: (N, in_dim)
Output Shape: (N, 3, 64, 64)
"""

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
        # print(x.shape)
        output = self.decoder(x)
        # print(output.shape)
        return output


class Discriminator(nn.Module):
    def __init__(self, in_dim=64):
        super(Discriminator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, in_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_dim, in_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_dim * 4, in_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.decoder(x)
        return output.view(-1, 1).squeeze(1) 

"""
For calculating Fretchet Distance
"""
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = torchvision.models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def get_transform(train=True):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

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


def calculate_activation_statistics(images,model,batch_size=128, dims=2048, cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value


def main():
    ###########################################################
    train_root = 'hw2_data/face/train'
    ###########################################################
    
    # hyperparameters
    batch_size = 128
    latent = 100
    num_epoch = 100
    lr = 0.0002

    train_dataset = MyDataset(root=train_root, transforms=get_transform(True))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=24,)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')
    print(f'img shape: {train_dataset[0][0].shape}, label shape: {train_dataset[0][1].shape}')

    G = Generator().to(device)
    G.apply(weights_init)
    D = Discriminator().to(device)
    D.apply(weights_init)
    print(G)
    print(D)
    
    # fid 
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    # loss function
    critertion = nn.BCELoss()
    
    # optimizer
    paramsG = [p for p in G.parameters() if p.requires_grad]
    optimizerG = torch.optim.Adam(paramsG, lr=lr, betas=(0.5, 0.999))
    paramsD = [p for p in D.parameters() if p.requires_grad]
    optimizerD = torch.optim.Adam(paramsD, lr=lr, betas=(0.5, 0.999))
    
    # scheduler
    schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=5, min_lr=0.00001, verbose=True)
    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor=0.5, patience=5, min_lr=0.00001, verbose=True)

    fixed_noise = torch.randn(1000, latent, 1, 1, device=device)
    # adding label smoothing
    real_ = 1.
    fake_ = 0.

    G_loss_epoch = []
    D_loss_epoch = []
    img_list = []
    best_fid = math.inf

    for epoch in range(0, num_epoch, 1):
        D_loss = []
        G_loss = []
        for i , img in enumerate(train_dataloader):
            #----------------------------------------Training D----------------------------------------

            for _ in range(2):
                D.zero_grad()

                # train with real img -> ground truth = real label
                real_img = img.to(device)
                real_size = real_img.size(0)

                # add some noise to the input to discriminator
                # real_img = 0.9*real_img + 0.1*torch.randn((real_img.size()), device=device)
                real_label = torch.full((real_size,), real_, dtype=torch.float, device=device)

                output = D(real_img).view(-1)
                errD_real = critertion(output, real_label)
                errD_real.backward()
                
                # train with fake img -> ground truth = fake label
                noise = torch.randn(real_size, latent, 1, 1, device=device)
                fake = G(noise)
                # add some noise to the input to discriminator
                # fake = 0.9*fake + 0.1*torch.randn((fake.size()), device=device)
                fake_label = torch.full((real_size,), fake_, dtype=torch.float, device=device)

                output = D(fake.detach()).view(-1)
                errD_fake = critertion(output, fake_label)
                errD_fake.backward()
                
                # update D
                errD = errD_real + errD_fake
                # errD.backward()
                optimizerD.step()

                if (errD == 0) or (math.isnan(errD)):
                    break
            #----------------------------------------Training G----------------------------------------
            G.zero_grad()
            fake_label_for_G = torch.full((real_size,), real_, dtype=torch.float, device=device)
            
            output = D(fake).view(-1)
            errG = critertion(output, fake_label_for_G)
            errG.backward()
            optimizerG.step()

            if (errG == 0) or (math.isnan(errG)):
                break

            D_loss.append(errD.item())
            G_loss.append(errG.item())

        current_D_loss = sum(D_loss)/len(D_loss)
        current_G_loss = sum(G_loss)/len(G_loss)

        D_loss_epoch.append(current_D_loss)
        G_loss_epoch.append(current_G_loss)
        fretchet_dist = calculate_fretchet(real_img, fake, model) 
        
        # print the proformace
        print(f'Epoch: [{epoch+1}/{num_epoch}], D Loss: {current_D_loss:.4f}, G Loss:{current_G_loss:.4f}, Fretchet_Distance: {fretchet_dist:.4f}')
        
        schedulerD.step(errD)
        schedulerG.step(errG)

        #evaluation
        fixed_img_output = G(fixed_noise)
        torchvision.utils.save_image(fixed_img_output.cpu().data[:32], f'output/epoch{epoch+1}.jpg', nrow=8, normalize=True)

        if fretchet_dist <= best_fid:
            best_fid = fretchet_dist
            torch.save({'epoch': epoch, 
                        'model_state_dict': G.state_dict(), 
                        'optimizer_state_dict': optimizerG.state_dict(),
                        'loss': errG,}, 'G_model_best.pth')
            print(f'Saving model with fid = {best_fid:.4f}')


    torch.save({'epoch': epoch, 
                        'model_state_dict': G.state_dict(), 
                        'optimizer_state_dict': optimizerG.state_dict(),
                        'loss': errG,}, 'G_model_final.pth')

    plot_loss_curve(D_loss_epoch, G_loss_epoch, num_epoch)


if __name__ == "__main__":
    main()