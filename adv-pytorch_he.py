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


import matplotlib.pyplot as plt

import dataloader as dl

from copy import deepcopy
import pickle as pkl
#%%

class Smallnet(nn.Module):
    
    def __init__(self,img_rows, img_cols, n_patients):
        super(Smallnet,self).__init__()
        
        self.n_patients = n_patients
        
        self.bn1 = nn.BatchNorm2d(32, )
        self.bn2 = nn.BatchNorm2d(64, )
        
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 32,
                               kernel_size = (3,3),
                               bias = False,
                               )
        
        t.nn.init.uniform_(self.conv1.weight,-1,1)
        
        
        self.conv2 = nn.Conv2d(in_channels = 32,
                               out_channels = 64,
                               kernel_size = (3,3),
                               bias = False,
                               )
        
        t.nn.init.uniform_(self.conv2.weight,0,1)
        
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = (2,2))
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(in_features = 23*23*64,
                             out_features = 128, 
                             )
        
        t.nn.init.uniform_(self.fc1.weight,-1,1)
        
        self.dropout2 = nn.Dropout(0.2)
        
        self.patient = nn.Linear(in_features = 128,
                                 out_features = n_patients,
                                 bias = False,
                                 )
        
        t.nn.init.uniform_(self.patient.weight,0,1)

        self.tumorclass = nn.Linear(in_features = 128,
                                    out_features = 2,
                                    bias = False,
                                    )
        
        t.nn.init.uniform_(self.tumorclass.weight,-1,1)
        
        
        
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
       
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = x.view(1,-1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        p = self.patient(x)
        y = self.tumorclass(x)
        
#        print(y)
        
        return p,y




def train(model,
          loss_function,
          data,
          optimizer,
          n_epochs = 100,
          ):

    if n_epochs == -1:
        n_epochs = np.inf
    
    epoch = 0
    
    best_model = deepcopy(model.state_dict())
    best_model_loss = np.inf
    
    n_train_samples = data.dataset.n_train_tiles
    n_val_samples = data.dataset.n_val_tiles
    
    loss = nn.CrossEntropyLoss()
 
#    model.conv2.register_forward_hook(printnorm)

    
    while epoch < n_epochs:
        
        t_loss_total = 0.0
        t_items = 0.0
        # TODO: adjust for np.inf which is not digit
        print(f'Epoch : {epoch:d}/{n_epochs:d}')
        
        model.train()
        
        
#        with t.set_grad_enabled(True):
        for sample in range(0,n_train_samples, n_batch):
            
            t_img, t_id, t_class = data.dataset.__getitem__(sample,'train')
            if t_img is not None:

                optimizer.zero_grad()
                
                t_id = label_encoder_patient.transform(np.array([t_id]).reshape(-1,1))
                
                t_id = Variable(t.LongTensor(t_id))
                
                t_class = Variable(t.LongTensor(np.array([t_class])))
            
                t_img = Variable(t_img.unsqueeze(0).float(),requires_grad = True)
                
                
                p_id, p_class = model(t_img)
                
                
                tmp_parameters = deepcopy(model.parameters)
                
                id_loss = -loss(input = p_id, target = t_id)
                
                id_loss.backward(retain_graph = True)
                model.patient.weight.grad = model.patient.weight.grad*(-1.0)
                
                class_loss = loss(input = p_class, target = t_class)
                class_loss.backward()

                optimizer.step()
                
                t_loss_total += id_loss.item() + class_loss.item()
                t_items += 1
        
        val = False
        if val:          
            with t.set_grad_enabled(False):
                v_loss_total = 0.0
                v_items = 0.0
                for sample in range(n_val_samples):
                    v_img, v_id, v_class = data.dataset.__getitem__(sample,'val')
                    v_id = hot_encoder_patient.transform(v_id)
                    v_img, v_id, v_class = v_img.to(device), v_id.to(device), v_class.to(device)
                    
                    if v_img is not None:
                        p_id, p_class = model(v_img)
                        s_loss = loss_function(true_id = v_id, predicted_id = p_id,
                                               true_class = v_class, predicted_class = p_class)
                        
                        v_loss_total += s_loss.item()
                        v_items += 1
                        
        print(f'training set loss : {t_loss_total/t_items:f}')
                    
        if t_loss_total < best_model_loss:
            best_model_loss = t_loss_total
            best_model = deepcopy(model.state_dict())
            with open('bestmodel.txt','w+') as fopen:
                fopen.write(best_model)
    

def joint_loss_function(true_id, predicted_id, true_class, predicted_class):
    loss = nn.CrossEntropyLoss()
    part1 = loss(input = predicted_id,target = true_id)
    part2 = loss(input = predicted_class, target = true_class)
#    print(true_id, predicted_id)
    return part1 


#%%
ROOT = '/home/alma/ST-2018/ST-2018/CNNp/data/breast-histopathology-images/IDC_regular_ps50_idx5'
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


FlipHorizontal = p_transforms.RandomHorizontalFlip(p = 0.5)
FlipVertical = p_transforms.RandomVerticalFlip(p = 0.5)
Crop = p_transforms.CenterCrop(50)

he_transforms = p_transforms.Compose([
                                     dl.Rotate(),
                                     dl.RandomHue(),
                                     dl.TransformToTensor(),
                                     ])

dataset = dl.HEDataset(root = ROOT, user_transform = he_transforms, num_patients = 4)
stacked = np.hstack((np.array(dataset.train_idx),np.array(dataset.val_idx))).reshape(-1,1)
label_encoder_patient = LabelEncoder()
label_encoder_patient.fit(stacked)

hot_encoder_patient = OneHotEncoder(sparse = False)
hot_encoder_patient.fit(label_encoder_patient.transform(stacked).reshape(-1,1))

print(dataset.n_train_tiles)

net = Smallnet(img_rows  = 50, img_cols = 50, n_patients = dataset.n_train + dataset.n_val)
dataset = t.utils.data.DataLoader(dataset = dataset)

opt = t.optim.Adagrad(net.parameters(),)

train(model = net,
      data = dataset,
      optimizer = opt,
      loss_function= loss_function,
      n_epochs = 10)

