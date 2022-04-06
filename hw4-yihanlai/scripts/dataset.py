import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# dataset
class MiniDatasetTrain(Dataset):
    def __init__(self, img_file, csv_file):
        super(MiniDatasetTrain, self).__init__()
        self.img_file = img_file
        self.csv_file = pd.read_csv(csv_file)
        self.ids = list(os.listdir(img_file))
        
        self.transform = transforms.Compose([
            # transforms.Resize(84),
            # transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        imgs = []
        for i in range(len(self.csv_file)):
            img_path = os.path.join(img_file, self.csv_file.iloc[i, 1])
            imgs.append(img_path)
        
        self.imgs = imgs
        self.csv_file.label = pd.Categorical(pd.factorize(self.csv_file.label)[0])
        self.labels = self.csv_file['label'].tolist()

    def __getitem__(self, index):
        path, label = self.imgs[index], self.labels[index]
        img = self.transform(Image.open(path).convert('RGB'))

        return img, label

    def __len__(self):
        return len(self.ids)


# Batch sampler
class GeneratorSamplerTrain():
    def __init__(self, labels, n_way, n_support, n_query, n_iter):
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_iter = n_iter

        labels = np.array(labels)
        self.k_shot = n_support + n_query
        self.cls = np.unique(labels)

        self.img_index = []
        for i in range(len(self.cls)):
            idxs = np.argwhere(labels == self.cls[i]).reshape(-1)
            idxs = torch.from_numpy(idxs)
            self.img_index.append(idxs)

    def __iter__(self):
        for i in range(self.n_iter):
            cls_index = torch.randperm(len(self.cls))[:self.n_way]
            batch = []
            for c in cls_index:
                imgs = self.img_index[c]
                sample_index = torch.randperm(len(imgs))[:self.k_shot]
                sampel = imgs[sample_index]
                batch.append(sampel)
            batch = torch.stack(batch).t().reshape(-1)
            # print(batch)
            yield batch    
    
    def __len__(self):
        return self.n_iter