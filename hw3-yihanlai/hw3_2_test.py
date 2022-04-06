import sys
import torch

from transformers import BertTokenizer
from PIL import Image

from models import caption
from datasets import coco, utils
from configuration import Config
import os

import matplotlib.pyplot as plt
import numpy as np


#############################################################
test_root = sys.argv[1]
output_path = sys.argv[2]
version = 'v3'
#############################################################

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
# print(model)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def evaluate(image_tensor):
    model.eval()
    count = 0
    for i in range(config.max_position_embeddings - 1):
        predictions, att_map = model(image_tensor, caption, cap_mask) #shape: (128, 30522)
        predictions = predictions[:, i, :] #shape: (30522)
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102: # end of sentence
            return caption, att_map
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        count += 1

    return caption, att_map


def visualize_attention_map(img_name, image, caption, att_map):
    res = [idx for idx, val in enumerate(caption) if val != 0]
    fig = plt.figure(figsize=(8,6))       
    for j in range(len(res)):
        img = np.asarray(image)
        att_map_each = Image.fromarray(att_map[j]).resize((img.shape[1],img.shape[0]))
        if len(res)%3 == 0 :
            plt.subplot(3, int(len(res)/3+1), j+1)
        if len(res)%3 == 1 :
            plt.subplot(3, int((len(res)+3)/3), j+1)
        else:
            plt.subplot(3, int((len(res)+3)/3), j+1)
        
        if j == 0:
            plt.imshow(image)
            plt.title('<start>')
            plt.axis('off')
            continue
        plt.imshow(image)
        plt.imshow(att_map_each/np.max(att_map_each), alpha=0.7, cmap='rainbow')
        plt.title(tokenizer.decode(caption[j]))
        if tokenizer.decode(caption[j]) == '.':
            plt.title('<end>')
        plt.axis('off')  
    fig.tight_layout(h_pad=0.3, w_pad=0.5) 
    plt.savefig(f"{output_path}/{img_name[:-4]}.png")


######### open a folder ##########
test_img = os.listdir(test_root) 
for img in test_img:
    print(f'Processing {img} ...') 
    imgPath = os.path.join(test_root, img)
    image = Image.open(imgPath)
    image_tensor = coco.val_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

    output, att_map = evaluate(image_tensor)
    visualize_attention_map(img,image, output[0].tolist(),att_map)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(result.capitalize())