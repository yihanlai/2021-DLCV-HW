import sys
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


def test(imgPath, resnet, classifier, encoder, device):
    custom_transforms = []
    custom_transforms.append(transforms.Resize((128, 128)))
    custom_transforms.append(transforms.ToTensor())
    custom_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_transforms = transforms.Compose(custom_transforms)
    
    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transforms(img).float()
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    feature = resnet(img_tensor)
    output = classifier(feature)

    pred = torch.argmax(output, dim=1).cpu().item()
    return pred
    

def main():
    #############################################################
    backbone = 'model_best_resnet.pth'
    classifier = 'model_best_classifier.pth'
    test_csv =  sys.argv[1] 
    test_data =  sys.argv[2]
    output_csv = sys.argv[3] 
    #############################################################

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint_backbone = torch.load(backbone)
    checkpoint_classifier = torch.load(classifier)
    
    resnet = models.resnet50(pretrained=False)
    resnet.load_state_dict(checkpoint_backbone["model_state_dict"])
    resnet.to (device)
    
    classifier = Classifier()
    classifier.load_state_dict(checkpoint_classifier["model_state_dict"])
    classifier.to(device)

    resnet.eval()
    classifier.eval()

    # test_img = os.listdir(test_data)
    # test_img.sort()
    df = pd.read_csv(test_csv)
    test_img = df['filename'].to_list()
    test_label = []

    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes.npy', allow_pickle=True)

    for i, img in enumerate(test_img):
        print(f'Processing {img} ...')
        imgPath = os.path.join(test_data, img)

        with torch.no_grad():
            label = test(imgPath, resnet, classifier, encoder, device)
            test_label.append(label)

    test_label = encoder.inverse_transform(test_label)
    ids = [i for i in range(len(test_label))]
    result = pd.DataFrame({'id': ids,'filename':test_img, 'label':test_label})
    result.to_csv(output_csv, index=False)

   
if __name__ == "__main__":
    main()