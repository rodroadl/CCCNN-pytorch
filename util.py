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
    def __init__(self, patch_size, num_patches, mask_coord=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_coord = mask_coord

    def __call__(self, img):
        print("Sampling Random Patches")
        if torch.is_tensor(img):
            _, h, w = img.size()
        else:
            h, w, _ = img.shape
        # left_upper, right_upper, right_lower, left_lower = self.mask_coord #(182,473)
        
        diameter = self.patch_size
        radius = self.patch_size // 2
        coords = set()
        center = list()

        for row in range(h):
            for col in range(w):
                if (row < h-radius-250 or col < w-radius-175): coords.add((row, col)) 

        for _ in range(self.num_patches):
            valid = False
            while not valid:
                y0, x0 = sample(coords, 1)[0]
                coords.remove((y0, x0))
                for y, x in center:
                    if not valid: break
                    valid &= abs(y-y0) > diameter and abs(x-x0) > diameter
                # termination: valid=False or (valid=True and i = len(taken))
            if valid: center.append((y0,x0))

        patches = []
        for x,y in center:
            if torch.is_tensor(img):
                patch = img[:, x-16:x+16, y-16:y+16]
            else:
                patch = img[x-16:x+16, y-16:y+16 , :]
            patches.append(patch)
        print("Done")
        return patches