#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 07:51:24 2018

@author: alma
"""
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import numpy as np
import pandas as pd

import torch as t
import torch.nn as nn
import torchvision.transforms as p_transforms
from torch.autograd import Variable


import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler


import matplotlib.pyplot as plt

#import dataloader as dl

from copy import deepcopy
import pickle as pkl
#%%

class Smallnet(nn.Module):
    
    def __init__(self,img_rows, img_cols,):
        super(Smallnet,self).__init__()
        
        
#        self.bn1 = nn.BatchNorm2d(32, )
#        self.bn2 = nn.BatchNorm2d(64, )
        
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 10,
                               kernel_size = (5,5),
                               bias = False,
                               )
        
        t.nn.init.uniform_(self.conv1.weight,-1,1)
        
        
        self.conv2 = nn.Conv2d(in_channels = 10,
                               out_channels = 20,
                               kernel_size = (5,5),
                               bias = False,
                               )
        
        t.nn.init.uniform_(self.conv2.weight,0,1)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.maxpool = nn.MaxPool2d(kernel_size = (2,2))
        
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(in_features = 320,
                             out_features = 50, 
                             )
        
        
        t.nn.init.uniform_(self.fc1.weight,-1,1)
        
        self.dropout2 = nn.Dropout(0.2)
        
        self.rounded = nn.Linear(in_features = 50,
                                 out_features = 2,
                                 bias = False,
                                 )
        
        t.nn.init.uniform_(self.rounded.weight,0,1)

        self.digit = nn.Linear(in_features = 50,
                                    out_features = 10,
                                    bias = False,
                                    )
        
        t.nn.init.uniform_(self.digit.weight,-1,1)
        
        
        
    def forward(self,x):
        
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.dropout1(x)
        x = x.view(x.shape[0],1,-1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        d = self.digit(x)
        r = self.rounded(x)
        
        
        
        return r,d


def train(model,
          loss_function,
          train_data,
          val_data,
          optimizer,
          n_epochs = 100,
          ):

    if n_epochs == -1:
        n_epochs = np.inf
    
    epoch = 0
    
    best_model = deepcopy(model.state_dict())
    best_model_loss = np.inf
    
    
    while epoch < n_epochs:
        
        r_loss_total = 0.0
        d_loss_total = 0.0
        
        # TODO: adjust for np.inf which is not digit
        if epoch % 2 == 0:             
            print(f'Epoch : {epoch:d}/{n_epochs:d}')
        
        model.train()
        
        loss = nn.CrossEntropyLoss()
        for minibatch in train_data:
            optimizer.zero_grad()

            images,t_digit = minibatch
            
            t_rounded = has_round_edges(t_digit)
            
            t_digit = Variable(t_digit.long())
            t_rounded = Variable(t_rounded)

            p_rounded, p_digit = model(images)
            # TODO : adjust for this in forward
            p_rounded = p_rounded.view(p_rounded.shape[0],2)
            p_digit = p_digit.view(p_digit.shape[0],10)
            
            
            rounded_loss = -loss(input = p_rounded, target = t_rounded)
            
            rounded_loss.backward(retain_graph = True)
            model.rounded.weight.grad *= -1.0

            
            digit_loss = loss(input = p_digit, target = t_digit)
            
            digit_loss.backward()

            optimizer.step()
            
            d_loss_total += digit_loss / images.shape[0]
            r_loss_total += rounded_loss / images.shape[0]
            
            
        t_loss_total = d_loss_total + r_loss_total
        
        epoch += 1
        
        # TODO : Add validation accuracy test after each epoch
        
        print(f'training set total_loss : {t_loss_total:f} | digit loss : {d_loss_total:f} | rounded loss : {r_loss_total:f} ')
                    
        if t_loss_total < best_model_loss:
            best_model_loss = t_loss_total
            best_model = deepcopy(model.state_dict())
            with open('bestmodel.txt','w+') as fopen:
                fopen.write(str(best_model))
    
#%%


def has_round_edges(numbers):
    
    rounded_numbers = [0,2,6,8,9]
    binary_output = [ int(x.item() in rounded_numbers) for x in numbers]
    binary_output = t.LongTensor(np.array(binary_output))
    
    return binary_output



#%%
ROOT = '/home/alma/ST-2018/ST-2018/CNNp/data/breast-histopathology-images/IDC_regular_ps50_idx5'
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

# TODO : Add CUDA distribution

p_split = dict(train = 0.8, val = 0.2)
batch_size = 64
num_workers = 0

net = Smallnet(28,28)    

train_set = datasets.MNIST(root='./data', train=True, download=False, transform=p_transforms.ToTensor())
val_set = datasets.MNIST(root='./data', train=True, download=False, transform=p_transforms.ToTensor())
split = int(len(train_set)*p_split['train'])

idx = np.arange(0,len(train_set))
np.random.seed(1001)
np.random.shuffle(idx)

train_idx, val_idx = idx[split:], idx[:split]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = t.utils.data.DataLoader(train_set,
                                       batch_size= batch_size,
                                       sampler = train_sampler,
                                       num_workers = num_workers,
                                       )


val_loader = t.utils.data.DataLoader(train_set,
                                     batch_size= batch_size,
                                     sampler = val_sampler,
                                     num_workers = num_workers,
                                     )


test_set = datasets.MNIST(root ='./data',train = False, download = False, transform = p_transforms.ToTensor())

test_loader = t.utils.data.DataLoader(test_set,
                                       batch_size= batch_size,
                                       num_workers = num_workers,
                                       )


opt = t.optim.SGD(net.parameters(),lr = 0.001, momentum=0.0)
loss_function = nn.CrossEntropyLoss()
#%%
train(model = net,
      train_data = train_loader,
      val_data = val_loader,
      optimizer = opt,
      loss_function= loss_function,
      n_epochs = 2500)

