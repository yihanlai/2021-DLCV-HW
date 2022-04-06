import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CosineSimilarity
import numpy as np


# CNN Feature Extractor - Conv-4
class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Proto_loss_acc(nn.Module):
    def __init__(self, n_way, n_support, n_query, mode, device):
        super().__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.mode = mode
        self.device = device

        if mode == "Euclidean":
            self.dist = euclidean_dist
        elif mode == "ConsineSimilarity":
            self.dist = consine_dist
        elif mode == "Parametric":
            self.dist = Parametric_func(n_way).to(device)

    def forward(self, query_feature, support_feature, label):
        # need to fix
        if type(label[0]) != str:
            label = [i.item() for i in label]

        # for k shot: calculate mean
        support_feature = support_feature.reshape(-1 ,self.n_way, support_feature.shape[-1])
        support_feature = support_feature.mean(axis=0)
        
        # for p1-3
        query_label = torch.arange(self.n_way).repeat(self.n_query)
        query_label = torch.LongTensor(query_label).to(self.device)
        # for p1-1, 1-2
        # label_encoder = {label[i * self.n_support] : i for i in range(self.n_way)}
        # query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in label[self.n_support * self.n_way:]])
        # print(query_label)

        dist = self.dist(query_feature, support_feature)
        loss = F.cross_entropy(dist, query_label)
        pred = torch.argmax(dist, dim=1)
        acc = (pred == query_label).float().mean().item()

        return loss, acc

class Parametric_func(nn.Module):
    def __init__(self, n_way):
        super().__init__()
        self.liner = nn.Sequential(
            nn.Linear(n_way*1600, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_way),
        )
    
    def forward(self, a, b):
        # print(f"qurey shape:　{a.shape}")
        # print(f"support shape: {b.shape}")
        
        n = a.shape[0] # query amount
        m = b.shape[0] # support amount
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)

        x = (a - b)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.liner(x)
        # print(x.shape)
        # exit()
        return x

def euclidean_dist(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = -((a - b)**2).sum(dim=2)

    return logits

def consine_dist(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    cos_sim = CosineSimilarity(dim=-1)
    logits = cos_sim(a, b)
    return logits