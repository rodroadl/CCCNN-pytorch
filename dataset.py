'''
dataset.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 12th, 2023
CS 7180: Advnaced Perception

Custom dataset
Resize as max(w,h) = 120 and sampling 32x32 patchs from the image as paper describe
'''

import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from util import read_16bit_png, MaxResize

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.images_dir = Path(data_dir)
        self.labels = pd.read_csv(label_file)
        self.images = os.listdir(self.images_dir)
        self.transform = transforms.Compose([
            MaxResize(1200),
            transforms.CenterCrop(32)]
        )


    def __getitem__(self, idx):
        image = read_16bit_png(os.path.join(self.images_dir,self.images[idx]))
        label = torch.tensor(self.labels.iloc[idx, 1:4].astype(float).values, dtype=torch.float32)
        if self.transform: image = self.transform(image)
        image = image.type(torch.float32) # necessary?
        return image, label
    
    def __len__(self):
        return len(self.images)