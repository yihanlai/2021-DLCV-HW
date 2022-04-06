import random
import os
import numpy as np
from PIL import Image
import pandas as pd
import sys

import torchvision

import torch
import torch.nn as nn


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



def test(imgPath, label_predictor, feature_extractor, device):
    
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    test_transform = torchvision.transforms.Compose(custom_transforms)

    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transform(img).float()
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    output = label_predictor(feature_extractor(img_tensor))
    pred = torch.argmax(output).cpu().detach().numpy()
    
    return pred


def main():
    ###################################################
    test_root = sys.argv[1]
    target_domain = sys.argv[2]
    output =  sys.argv[3] 
    ###################################################
    label_predictor_weight = f'{target_domain}_label_predictor.pth'
    feature_extractor_weight = f'{target_domain}_extractor.pth'
    
    print(f'###Testing on target domain : {target_domain}###')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    label_predictor = LabelPredictor().to(device)
    checkpoint = torch.load(label_predictor_weight)
    label_predictor.load_state_dict(checkpoint['model_state_dict'])
    label_predictor.eval()

    feature_extractor = FeatureExtractor().to(device)
    checkpoint = torch.load(feature_extractor_weight)
    feature_extractor.load_state_dict(checkpoint['model_state_dict'])
    feature_extractor.eval()

    test_img = os.listdir(test_root)
    test_img.sort()
    pred_label = []

    for i, img in enumerate(test_img):
        print(f'Processing {img} ...')
        imgPath = os.path.join(test_root, img)

        with torch.no_grad():
            label = test(imgPath, label_predictor, feature_extractor, device)
            pred_label.append(label)

    result = pd.DataFrame({'image_name':test_img, 'label':pred_label})
    result.to_csv(output, index=False)

    print("That's it!")
    

if __name__ == "__main__":
    main()