import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from PIL import Image
import pandas as pd

import sys


def get_pretrained_model(num_classes):
    # load the model pre-trained on COCO
    model = torchvision.models.vgg16_bn(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def test(imgPath, model, device):
    
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize(size=(224, 224)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    test_transform = torchvision.transforms.Compose(custom_transforms)

    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transform(img).float()
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    output = nn.functional.softmax(model(img_tensor), dim=1).data.cpu().numpy()[0]
    pred = np.argmax(output)
    
    return pred


def main():
    #############################################################
    weight = 'model_best_vgg16_with_bn.pth'
    test_root =  sys.argv[1] # 'p1_data/val_50'
    output_root = sys.argv[2] # 'p1_data/' 
    #############################################################

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')

    num_classes = 50
    model = get_pretrained_model(num_classes)
    model.to(device)
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_img = os.listdir(test_root)
    test_img.sort()
    test_label = []
    truth_lable = []

    for i, img in enumerate(test_img):
        print(f'Processing {img} ...')
        imgPath = os.path.join(test_root, img)
        
        with torch.no_grad():
            label = test(imgPath, model, device)
            test_label.append(label)
            truth_lable.append(int(img.split('_')[0]))
    
    result = pd.DataFrame({'image_id':test_img, 'label':test_label})
    result.to_csv(output_root, index=False)

    acc = np.mean(np.array(test_label)==np.array(truth_lable))
    print(f"Validation/Test Accuracy={acc}")
    print("That's it!")
    
if __name__ == "__main__":
    main()