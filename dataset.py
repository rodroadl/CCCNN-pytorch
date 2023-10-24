'''
dataset.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 20th, 2023
CS 7180: Advnaced Perception

Custom dataset
Transform by Contrast Normalization - Global Histogram Stretching - 
and Randomly sample 32 by 32 patches
'''

import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from util import read_16bit_png, ContrastNormalization, RandomPatches, MaxResize

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file, num_patches, image_space='linear', label_space='linear'):
        '''
        constructor

        Parameters:
            data_dir(str or Path) - path for directory containing images
            lable_file(str or Path) - path for label file
            num_patches(int) - Number of patches that gets to pass into RandomPatches
            log_space(bool, optional) - Flag whether to map chromaticity space to log chromaticity space
        '''
        self.images_dir = Path(data_dir)
        self.labels = pd.read_csv(label_file)
        self.images = os.listdir(self.images_dir)
        self.num_patches = num_patches
        self.image_space = image_space
        self.label_space = label_space
        self.transform = transforms.Compose([
                # MaxResize(1200), # SimpleCube++ has width of 648 and height of 432
                ContrastNormalization(),
                RandomPatches(patch_size = 32, num_patches = self.num_patches)
                ])
    def __getitem__(self, idx):
        '''
        Return an images and labels for given index

        Parameters:
            idx(int) - index

        Return:
            image(sequence of tensors)
            label(sequence of tensors)
        '''
        image = read_16bit_png(os.path.join(self.images_dir,self.images[idx]))
        label = torch.tensor(self.labels.iloc[idx, 1:4].astype(float).values, dtype=torch.float32) 

        # find saturation level for expanded log space
        if self.image_space == 'expandedLog' or self.label_space == 'expandedLog': saturation_lvl = torch.max(image)
        else: eps = 1e-7
        
        # transform
        if self.transform: image = self.transform(image)
        if self.image_space == 'log': # ->[-infty, 0]
            image = torch.log(image+eps)
        elif self.image_space == 'expandedLog': # ->[0, ~9.7]
            image *= saturation_lvl
            image[image != 0] = torch.log(image[image != 0])
        if self.label_space == 'log': # ->[-infty, 0]
            label = torch.log(label+eps)
        elif self.label_space == 'expandedLog': # ->[0, ~9.7]
            label *= saturation_lvl
            label[label != 0] = torch.log(label[label != 0])
            label = torch.clip(label, 0, saturation_lvl)

        return image, torch.stack([label] * image.shape[0], dim=0)
    
    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        return len(self.images)

class ReferenceDataset(Dataset):
    def __init__(self, data_dir, label_file):
        '''
        Constructor

        Parameters:
            data_dir(str or Path) - path for directory containing images
            lable_file(str or Path) - path for label file
        '''
        self.images_dir = Path(data_dir)
        self.labels = pd.read_csv(label_file)
        self.images = os.listdir(self.images_dir)

    def __getitem__(self, idx):
        '''
        Return an image and label for given index

        Parameters:
            idx(int) - index

        Return:
            image(tensor)
            label(tensos)
        '''
        image = read_16bit_png(os.path.join(self.images_dir,self.images[idx]))
        label = torch.tensor(self.labels.iloc[idx, 1:4].astype(float).values, dtype=torch.float32) 
        
        return image, label
    
    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        return len(self.images)
