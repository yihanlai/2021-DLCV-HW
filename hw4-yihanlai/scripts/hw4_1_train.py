import os
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from model import Convnet, Proto_loss_acc
from dataset import MiniDatasetTrain, GeneratorSamplerTrain
from test_testcase import MiniDataset, GeneratorSampler

# fix random seeds for reproducibility
SEED = 999
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    #########################################
    train_img = 'hw4_data/mini/train'
    train_csv = 'hw4_data/mini/train.csv'
    val_img = 'hw4_data/mini/val'
    val_csv = 'hw4_data/mini/val.csv'
    #########################################
    # hyperparameter
    num_epoch = 50
    lr = 0.0001
    n_train_way = 5
    n_test_way = 5
    n_support = 10 # k_shot
    n_query= 15
    n_iter = 600
    dist_mode = "Euclidean"

    # load data 
    train_dataset = MiniDatasetTrain(img_file=train_img, csv_file=train_csv)
    val_dataset = MiniDataset(val_csv,val_img)

    train_sampler = GeneratorSamplerTrain(train_dataset.labels, n_train_way, n_support, n_query, n_iter)
    val_sampler = GeneratorSampler('hw4_data/mini/val_testcase.csv')

    train_dataloader = DataLoader(train_dataset,
                            batch_sampler=train_sampler,
                            worker_init_fn=worker_init_fn,
                            num_workers=24,)

    val_dataloader = DataLoader(val_dataset,
                        sampler=val_sampler,
                        batch_size= n_test_way * (n_query + n_support),
                        worker_init_fn=worker_init_fn,
                        num_workers=24,)
    
    # for p1-3 validation
    val_dataset = MiniDatasetTrain(img_file=val_img, csv_file=val_csv)
    val_sampler = GeneratorSamplerTrain(val_dataset.labels, n_test_way, n_support, n_query, n_iter)
    val_dataloader = DataLoader(val_dataset,
                        batch_sampler=val_sampler,
                        worker_init_fn=worker_init_fn,
                        num_workers=24,)

    # define model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Convnet().to(device)

    # loss function
    proto_train = Proto_loss_acc(n_train_way, n_support, n_query, dist_mode, device)
    proto_test = Proto_loss_acc(n_test_way, n_support, n_query, dist_mode, device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(num_epoch):
        # ---------------------------------Training-------------------------------------
        model.train()
        train_loss = []
        train_acc = []
        
        for i, batch in enumerate(train_dataloader):
            data, label = batch
            p = n_support * n_train_way
            data_support, data_query = data[:p], data[p:]

            proto = model(data_support.to(device))
            query_feature = model(data_query.to(device))  
            
            loss, acc = proto_train(query_feature, proto, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc)
        
        # std = np.std(train_acc)
        # mean = np.mean(train_acc)
        # print(f"95% CI: ({mean-1.96*std/(600**0.5)}, {mean+1.96*std/(600**0.5)})")
        
        current_train_loss = sum(train_loss)/len(train_loss)
        current_train_acc = np.mean(train_acc)*100
        
        # ---------------------------------Validation-----------------------------------
        model.eval()
        val_loss = []
        val_acc = []

        for i, batch in enumerate(val_dataloader):
            data, label = batch
            p = n_support * n_test_way
            data_support, data_query = data[:p], data[p:]
            
            proto = model(data_support.to(device))
            query_feature = model(data_query.to(device))       
            
            loss, acc = proto_test(query_feature, proto, label)

            val_loss.append(loss.item())
            val_acc.append(acc)

        current_val_loss = sum(val_loss)/len(val_loss)
        current_val_acc = np.mean(val_acc)*100

        # print the performance
        print(f'Epoch: {epoch}/{num_epoch}, Training Loss: {current_train_loss:.4f}, Training Acc: {current_train_acc:.4f}, Validation Loss: {current_val_loss:.4f}, Validation Acc: {current_val_acc:.4f}')
        
        # update the model if the accuracy on the validation has improved
        if current_val_acc >=  best_acc:
            best_acc = current_val_acc
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,}, 'model_best.pth')
            print(f'Saving model with acc {best_acc:.4f}')

    print("That's it!")

if __name__ == "__main__":
    main()

    