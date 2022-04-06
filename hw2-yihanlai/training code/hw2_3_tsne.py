import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from PIL import Image
import pandas as pd

import sys
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import random

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
        # print(x.shape)
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
        c = self.layer(h)
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
        y = self.layer(h)
        return y


def get_features(total_img_path, label_predictor, feature_extractor, device):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    test_transform = torchvision.transforms.Compose(custom_transforms)

    labels = []
    features = None
    
    with torch.no_grad():
        for i, imgPath in enumerate(total_img_path):
            img = Image.open(imgPath).convert('RGB')
            img_tensor = test_transform(img).float()
            img_tensor = img_tensor.unsqueeze_(0).to(device)
            output = label_predictor(feature_extractor(img_tensor))
            pred = torch.argmax(output).cpu().detach().numpy()
            labels.append(pred)

            feat_list = []
            def hook(module, input, output):
                feat_list.append(output)

            handel = feature_extractor.register_forward_hook(hook)
            output = feature_extractor(img_tensor)
            feat = feat_list[0].unsqueeze_(0)
            handel.remove()

            current_features = feat.data.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features

    return labels, features


def get_domain(total_img_path, label_predictor, feature_extractor, device):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    test_transform = torchvision.transforms.Compose(custom_transforms)

    labels = []
    features = None
    
    with torch.no_grad():
        for i, imgPath in enumerate(total_img_path):
            img = Image.open(imgPath).convert('RGB')

            if i < 2000:
                pred = 1 # source domain= 1
            else:
                pred = 0 # target domain = 0
            
            img_tensor = test_transform(img).float()
            img_tensor = img_tensor.unsqueeze_(0).to(device)
            labels.append(pred)

            feat_list = []
            def hook(module, input, output):
                feat_list.append(output)

            handel = feature_extractor.register_forward_hook(hook)
            output = feature_extractor(img_tensor)
            feat = feat_list[0].unsqueeze_(0) 
            print()
            handel.remove()

            current_features = feat.data.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features

    return labels, features


def main():
    data = [ 'usps', 'mnistm', 'svhn', 'usps']
    model = ['mnistm2usps', 'svhn2mnistm', 'usps2svhn']

    for i in range(3):
        #############################################################
        source_root = f'hw2_data/digits/{data[i+1]}/test/' 
        target_root = f'hw2_data/digits/{data[i]}/test/'
        label_predictor_weight = f'model/{model[i]}_label_predictor.pth'
        feature_extractor_weight = f'model/{model[i]}_extractor.pth'
        domain_classifier_weight = f'model/{model[i]}_domain_classifier.pth'
        #############################################################
        print(f'##### DANN model: {model[i]} / source domain: {data[i+1]} / target domain: {data[i]} #####')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # load models
        label_predictor = LabelPredictor().to(device)
        checkpoint = torch.load(label_predictor_weight)
        label_predictor.load_state_dict(checkpoint['model_state_dict'])
        label_predictor.eval()

        feature_extractor = FeatureExtractor().to(device)
        checkpoint = torch.load(feature_extractor_weight)
        feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        feature_extractor.eval()

        domain_classifier = DomainClassifier().to(device)
        checkpoint = torch.load(domain_classifier_weight)
        domain_classifier.load_state_dict(checkpoint['model_state_dict'])
        domain_classifier.eval()

        # random select some img to test
        num_imgs = 2000
        s_img = os.listdir(source_root)
        s_img = random.sample(s_img, num_imgs)
        t_img = os.listdir(target_root)
        t_img = random.sample(t_img, num_imgs)
        # combine source and target domain images
        s_img_path = [source_root + i for i in s_img]
        t_img_path = [target_root + i for i in t_img]
        total_img_path = s_img_path + t_img_path


        # tsne feature part 
        labels, features = get_features(total_img_path, label_predictor, feature_extractor, device)
        labels = np.array(labels)

        tsne = TSNE(n_components=2).fit_transform(features)
        x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
        tsne = (tsne - x_min) / (x_max - x_min)

        plt.figure()
        cmap = plt.cm.get_cmap("tab10", 10)
        for index in range(10):
            indices = [i for i, l in enumerate(labels) if l == index]
            tx = np.take(tsne[:, 0], indices)
            ty = np.take(tsne[:, 1], indices)
            color = cmap(index)
            for x,y in zip(tx, ty):
                plt.text(x, y, str(index), color=color)
        plt.title(f"t-SNE embedding [{model[i]}]")
        plt.savefig(f'tsne_{model[i]}.png')


        # tsne domain part 
        labels, features = get_domain(total_img_path, label_predictor, feature_extractor, device)
        labels = np.array(labels)

        tsne = TSNE(n_components=2).fit_transform(features)
        x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
        tsne = (tsne - x_min) / (x_max - x_min)

        plt.figure()
        cmap = plt.cm.get_cmap("tab10",2)
        for index in range(2):
            indices = [i for i, l in enumerate(labels) if l == index]
            tx = np.take(tsne[:, 0], indices)
            ty = np.take(tsne[:, 1], indices)
            color = cmap(index)
            for x,y in zip(tx, ty):
                plt.text(x, y, str(index), color=color)
        plt.title(f"t-SNE embedding [{model[i]}]")
        plt.savefig(f'tsne_domain_{model[i]}.png')
 
 
    
if __name__ == "__main__":
    main()