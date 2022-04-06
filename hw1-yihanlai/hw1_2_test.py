import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F

from PIL import Image

import sys

import matplotlib.pyplot as plt
from model import VGG16_FCN32, VGG16_FCN8, UNet



def test(imgPath, model, device):
    
    cls_color = {
        0:  [0, 255, 255],
        1:  [255, 255, 0],
        2:  [255, 0, 255],
        3:  [0, 255, 0],
        4:  [0, 0, 255],
        5:  [255, 255, 255],
        6: [0, 0, 0],
        }

    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_transform = torchvision.transforms.Compose(custom_transforms)

    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transform(img).float()
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    
    output = F.softmax(model(img_tensor), dim=1)
    output = torch.argmax(output, dim=1).cpu().numpy()[0]

    # class to mask color
    mask = np.empty((512,512,3))
    for label in range(6):
        mask[output == label] = cls_color[label]
    # print(mask)

    return mask

def main():
    #############################################################
    test_root = sys.argv[1]
    output = sys.argv[2]
    weight = 'model_best_FCN8.pth'
    #############################################################

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')

    model = VGG16_FCN8(7)
    model.to(device)
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
 
    # test_img = list(sorted(f for f in os.listdir(test_root) if f.endswith('sat.jpg')))
    test_img = list(sorted(f for f in os.listdir(test_root)))

    for i, img in enumerate(test_img):
        print(f'Processing {img} ...')
        imgPath = os.path.join(test_root, img)
        
        with torch.no_grad():
            mask = test(imgPath, model, device)
            mask = Image.fromarray(np.uint8(mask))
            mask.save(f'{output}/{img[:4]}.png')


    print("That's it!")
    
if __name__ == "__main__":
    main()