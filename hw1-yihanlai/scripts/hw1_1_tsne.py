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


def get_pretrained_model(num_classes):
    # load the model pre-trained on COCO
    model = torchvision.models.vgg16_bn(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def get_features(test_root, test_img, model, device):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(size=(224, 224)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    test_transform = torchvision.transforms.Compose(custom_transforms)
    
    labels = []
    features = None
    
    with torch.no_grad():
        for i, img in enumerate(test_img):
            imgPath = os.path.join(test_root, img)

            img = Image.open(imgPath).convert('RGB')
            img_tensor = test_transform(img).float()
            img_tensor = img_tensor.unsqueeze_(0).to(device)
            output = nn.functional.softmax(model(img_tensor), dim=1).data.cpu().numpy()[0]
            pred = np.argmax(output)
            labels.append(pred)

            feat_list = []
            def hook(module, input, output):
                feat_list.append(output)

            handel = model.classifier[-4].register_forward_hook(hook)
            output = model(img_tensor)            
            feat = torch.flatten(feat_list[0], 1)
            handel.remove()

            current_features = feat.data.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features

    return labels, features

def main():
    #############################################################
    test_root = 'p1_data/val_50' #sys.argv[1]
    weight = '1_models/model_best_vgg16_with_bn.pth'
    #############################################################

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')

    num_classes = 50
    model = get_pretrained_model(num_classes)
    model.to(device)
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(model)
    
    test_img = os.listdir(test_root)
    test_img.sort()
    test_label = []
           
    # tsne part 
    labels, features = get_features(test_root, test_img, model, device)
    labels = np.array(labels)

    tsne = TSNE(n_components=2).fit_transform(features)
    x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
    tsne = (tsne - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(20,16))
    cmap = plt.cm.get_cmap("Set1", 50)
    for index in range(50):
        indices = [i for i, l in enumerate(labels) if l == index]
        tx = np.take(tsne[:, 0], indices)
        ty = np.take(tsne[:, 1], indices)
        color = cmap(index)
        for x,y in zip(tx, ty):
            plt.text(x, y, str(index), color=color)
        # plt.scatter(tx, ty, color=color,label=index)
    plt.title("t-SNE embedding")
    plt.savefig('tsne.png')
 
    
if __name__ == "__main__":
    main()