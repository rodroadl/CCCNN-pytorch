import math 
from random import randint, sample

import torch
from torchvision.io import read_file
from torchvision.transforms import functional as F

def read_16bit_png(file):
    data = read_file(file)
    return torch.ops.image.decode_png(data, 0, True)

def angularLoss(xs, ys):
    output = 0
    for x, y in zip(xs, ys):
        output += math.degrees(torch.arccos(torch.nn.functional.cosine_similarity(x,y, dim=0)).item())
    return output

def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    return torch.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)

def linearize_expand(img, black_lvl=0, saturation_lvl=2**16-1):
    return linearize(img, black_lvl, saturation_lvl) * (saturation_lvl - black_lvl)

def linearize_expand_log(img, black_lvl=0, saturation_lvl=2**16-1):
    x = torch.log(linearize_expand(img, black_lvl, saturation_lvl))
    x = torch.nn.functional.normalize(x)
    return torch.nan_to_num(x)

### Transform 

class MaxResize:
    def __init__(self, max_length):
        self.max_length = max_length
        
    def __call__(self, img):
        _, w, h = img.size()
        ratio = float(w) / float(h)
        if ratio > 1: # w > h
            h0 = math.ceil(self.max_length / ratio)
            return F.resize(img, (self.max_length, h0), antialias=True)
        else: # h <= w
            w0 = math.ceil(self.max_length / ratio)
            return F.resize(img, (w0, self.max_length), antialias=True)

class ContrastNormalization: # Contrast normalization - Global Histogram stretching
    def __init__(self, black_lvl=2048):
        self.black_lvl = black_lvl
    def __call__(self, img):
        saturation_lvl = torch.max(img)
        return (img - self.black_lvl)/(saturation_lvl - self.black_lvl)

class RandomPatches:
    def __init__(self, patch_size, num_patches):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, img):
        MASK_HEIGHT = 250
        MASK_WIDTH = 175
        _, h, w = img.size()        
        diameter = self.patch_size
        radius = self.patch_size // 2
        coords = set()
        center = list()

        for row in range(radius, h-radius):
            for col in range(radius, w-radius):
                if (row < h-radius-MASK_HEIGHT or col < w-radius-MASK_WIDTH): coords.add((row, col)) 

        for _ in range(self.num_patches):
            valid = False
            while coords and not valid:
                y0, x0 = sample(coords, 1)[0]
                coords.remove((y0, x0))
                valid = True
                for y, x in center:
                    if not valid: break
                    valid &= abs(y-y0) > diameter and abs(x-x0) > diameter
                # termination: valid=False or (valid=True and i = len(taken))
            if valid: center.append((y0,x0))

        patches = []
        for y,x in center:
            patch = img[:, y-16:y+16, x-16:x+16].type(torch.float32)
            patches.append(patch)
        
        t = torch.Tensor(len(patches), 32, 32)
        torch.cat(patches, out=t)
        return t