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
from util import read_16bit_png, MaxResize, ContrastNormalization, RandomPatches

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file, num_patches, log_space=False):
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
        self.log_space = log_space
        self.num_patches = num_patches
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
        
        if self.transform: image = self.transform(image)
        if self.log_space:
            max_val = 65535
            eps = 1e-7
            image[image != 0] = torch.log(max_val * image[image != 0])
            label = torch.log(label+eps)
            # image, label = torch.log(image+1e-7), torch.log(label+1e-7)

        return image, torch.stack([label] * image.shape[0], dim=0)
    
    def __len__(self):
        '''
        Return the length of the dataset

        Return:
            length(int)
        '''
        return len(self.images)
