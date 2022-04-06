import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision


from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import sys
import timm


def test(imgPath, model, device):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize((384, 384)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    test_transform = torchvision.transforms.Compose(custom_transforms)

    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transform(img).float()
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    
    output = nn.functional.softmax(model(img_tensor)).data.cpu().numpy()[0]
    pred = np.argmax(output)
    
    return pred


def position_embedding_similarities(model):
    pos_embed = model.pos_embed
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Visualization of position embedding similarities", fontsize=24)
    for i in range(1, pos_embed.shape[1]):
        sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((24, 24)).detach().cpu().numpy()
        ax = fig.add_subplot(24, 24, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.savefig('position embeddings.png')


def attention_map(imgName, imgPath, model, device):
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.Resize((384, 384)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    test_transform = torchvision.transforms.Compose(custom_transforms)
    
    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transform(img)
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    
    patches = model.patch_embed(img_tensor)
    pos_embed = model.pos_embed
    transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed # (class token + patches) + positon embeddings 
    
    # print("Transformer input: ", transformer_input.shape)
    attention = model.blocks[-1].attn # last mutil-head attention layer
    transformer_input_expanded = attention.qkv(transformer_input)
    # print("expanded to: ", transformer_input_expanded.shape)
    qkv = transformer_input_expanded.reshape(577, 3, 12, 64)
    q = qkv[:, 0].permute(1, 0, 2)  
    k = qkv[:, 1].permute(1, 0, 2)  
    kT = k.permute(0, 2, 1)  
    # print('q & k shape: ', q.shape,)
    attention_matrix = np.matmul(q.detach().cpu().numpy(), kT.detach().cpu().numpy())
    # print("attention matrix:", attention_matrix.shape)
    mean_attention_matrix = attention_matrix.mean(0)[0] # take class token as query
    # print("mean attention matrix: ", mean_attention_matrix.shape)
    attention_map = mean_attention_matrix[1:].reshape((24,24))
    attention_map = Image.fromarray(attention_map).resize((384,384))
    
    fig = plt.figure(figsize=(8, 8))
    img = np.asarray(img.resize((384,384)))
    plt.imshow(img)
    plt.imshow(attention_map/np.max(attention_map), alpha=0.6, cmap='rainbow')
    plt.savefig(f"attention_map_{imgName}")


def main():
    #############################################################
    test_root = sys.argv[1]
    output = sys.argv[2]
    #############################################################
    weight = 'model_best_hw3.pth'
    pics = ['26_5064.jpg','29_4718.jpg','31_4838.jpg']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=37)
    model.to(device)
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # print(model)

    test_img = os.listdir(test_root)
    test_img.sort()
    test_label = []
    truth_lable = []

    # Visualize position embedding similarities
    # position_embedding_similarities(model)

    # Visualize attention map
    # for img in pics:
        # imgPath = os.path.join(test_root, img)
        # attention_map(img, imgPath, model, device) 

    for i, img in enumerate(test_img):
        print(f'Processing {img} ...')
        imgPath = os.path.join(test_root, img)

        with torch.no_grad():
            label = test(imgPath, model, device)
            test_label.append(label)
            truth_lable.append(int(img.split('_')[0]))

    result = pd.DataFrame({'filename':test_img, 'label':test_label})
    result.to_csv(output,index=False)

    acc = np.mean(np.array(test_label)==np.array(truth_lable))
    print(f"Validation Accuracy={acc}")
    print("That's it!")
    
if __name__ == "__main__":
    main()