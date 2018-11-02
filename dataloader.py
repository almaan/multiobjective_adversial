#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 07:53:43 2018

@author: alma
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import numpy as np
from PIL import Image
import PIL

from HENorm import convert_image

import pandas as pd
import os.path as osp
import os


import sys

class HEDataset(Dataset):
    """
    For classification of HE-images.
    
    """
    
    def __init__(self, root,
                 num_patients = -1,
                 user_transform = None,
                 p = dict(test = 0.2, train = 0.8, val = 0.2)
                 ):
        """
        load paths for items, does not load images into memory.
        
        """
        self.root = root
        patient_idx = os.listdir(root)
        
        
        if num_patients > 0:
            print(num_patients)
            subset = np.random.choice(np.arange(0,len(patient_idx)), size = num_patients, replace = False )
            print(subset)
            patient_idx = [patient_idx[x] for x in subset]
        
        # TODO: change to self.patient_idx everywhere
        self.patient_idx = patient_idx
        
        idx = np.arange(0,len(patient_idx))
#        np.random.seed(1337) #for reprodcibility, remove later!
        np.random.shuffle(idx)
        
        self.test_idx = [patient_idx[x] for x in range(int(p['test']*len(idx)))] 
        self.n_test = len(self.test_idx)
        self.val_idx = [patient_idx[x] for x in range(self.n_test, self.n_test + int(len(idx)*(1.0 - p['test'])*p['val']))]
        self.n_val = len(self.val_idx)
        self.train_idx = [patient_idx[x] for x in range(self.n_test + self.n_val,len(idx))]
        self.n_train = len(self.train_idx)
        
        self.test_tiles = self._get_tile_names(self.test_idx)
        self.test_tiles = self._random_shuffle(self.test_tiles)
        self.val_tiles = self._get_tile_names(self.val_idx)
        self.val_tiles = self._random_shuffle(self.val_tiles)
        self.train_tiles = self._get_tile_names(self.train_idx)    
        self.train_tiles = self._random_shuffle(self.train_tiles)
        
        self.n_test_tiles = len(self.test_tiles)
        self.n_val_tiles = len(self.val_tiles)
        self.n_train_tiles = len(self.train_tiles)
        
        if user_transform is not None:
            self.transform = user_transform
        else:
            self.transform = TransformToTensor()
        
    def __getitem__(self, idx, dataset):
        
        
        dataset = dataset.lower()
        
        if dataset not in ['test','train','val']:
            raise ValueError('Invalid dataset specified, give test, train or val')
        
        use_set = eval(''.join(['self.',dataset,'_tiles']))
        
        patient_id, bc_class = self._get_name_class(use_set[idx])
        
        
        
        img_pth = osp.join(self.root,patient_id,bc_class,use_set[idx])
        
        img = Image.open(img_pth)
        
        if img.size[0] == img.size[1] and img.size[0] == 50:
        
            img = img.convert('RGB')    
        
            img = self.transform(img)
            
            patient_id = eval(patient_id)
            bc_class = eval(bc_class)
            
            return (img, patient_id, bc_class)
        else:
            return (None, None, None)
    
    def __len__(self,):
        
        return len(self.test_tiles) + len(self.train_tiles) + len(self.val_tiles)
        
    
    def _random_shuffle(self,listitem):
        
        N = len(listitem)
        idx = np.arange(N)
        np.random.shuffle(idx)
        
        listitem = [listitem[x] for x in idx]
        
        return listitem
    
    def _get_tile_names(self,dataset):
        tiles = []            
        for patient in dataset:
            for bc_type in ['0','1']:
                tmp = os.listdir(osp.join(self.root,patient,bc_type))
                tiles += tmp
        
        return tiles
     
    def _get_name_class(self,sample):
        
        raw_split = sample.split('_')
        name, bc_class = raw_split[0],raw_split[-1]
        bc_class = bc_class.split('.')[0][-1]
        
        return name, bc_class
            
class TransformToTensor(object):
    """
    Convert numpy array into tensor object
    
    """
    def __init__(self,):
        pass
    def __call__(self,image):

        if isinstance(image, Image.Image):
            image = np.array(image)
        elif not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except:
                print('image cannot be converted to numpy array')
       
        image = image.reshape((image.shape[-1],image.shape[0],image.shape[1]))
            
        
            #Pytorch uses C x H x W 
        out = torch.from_numpy(image)
        out = out.float()
        return out
    
class Rotate(object):
    """
    Randomly rotate image by any of the vales in the set {0,90,180,270} degrees
    
    """
    def __init__(self,):
        pass
    
    def __call__(self,image):
        k = np.random.randint(0,4)
        if k > 0:
            image = image.rotate(90*k)
        return image
#%%

class RandomHue(object):
    """
    Add small random noise of purple and pinkt tint to images
    
    
    """
    def __init__(self,):
        pass
    def __call__(self,image):
        h,w = image.size
        purple = (131,131,223)
        pink = (205,131,223)
        
        purple_mat = np.tile(purple,(h,w,1))
        purple_mat = purple_mat.reshape((h,w,3))
        pink_mat = np.tile(pink,(h,w,1))
        pink_mat = pink_mat.reshape((h,w,3))
        
        purple_mat = Image.fromarray((purple_mat).astype('uint8')).convert('RGBA')
        pink_mat = Image.fromarray((pink_mat).astype('uint8')).convert('RGBA')
        
        
        ratio = np.random.random()
        level = 50
        
        purple_mat.putalpha(int(ratio*level))
        pink_mat.putalpha(int((1.-ratio)*level))
        
        
        full_random = Image.alpha_composite(purple_mat,pink_mat)
        
        image = Image.alpha_composite(image.convert('RGBA'),full_random) 
        image = image.convert('RGB')
        return image

class Normalizer:
    """
    Use Macenko stain exctracion to normalize images
    
    """
    def __init__(self,):
        pass
    def __call__(self,image):
        image = convert_image(target_img = image)
    
        return image
        
    

if __name__ == '__main__':
    test_pth = '/home/alma/ST-2018/ST-2018/CNNp/data/breast-histopathology-images/IDC_regular_ps50_idx5'
    user_transform = RandomHue()
    data = HEDataset(test_pth,)
    v,b,c =  data.__getitem__(int(np.random.random()*100),'train')
    import matplotlib.pyplot as plt
    plt.imshow(v.numpy().astype('uint8'))
 