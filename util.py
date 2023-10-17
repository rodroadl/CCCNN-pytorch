import math 
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

def linearize(img, black_lvl=0, saturation_lvl=2**16-1):
    return torch.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)

def linearize_expand(img, black_lvl=0, saturation_lvl=2**16-1):
    return linearize(img, black_lvl, saturation_lvl) * (saturation_lvl - black_lvl)

def linearize_expand_log(img, black_lvl=0, saturation_lvl=2**16-1):
    x = torch.log(linearize_expand(img, black_lvl, saturation_lvl))
    x = torch.nn.functional.normalize(x)
    return torch.nan_to_num(x)

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

class Linearize:
    def __init__(self, black_lvl=2048, saturation_lvl=2**14-1):
        self.black_lvl = black_lvl
        self.saturation_lvl = saturation_lvl
    def __call__(self, img):
        return torch.clip((img - self.black_lvl)/(self.saturation_lvl - self.black_lvl), 0, 1)

class RandomPatches:
    def __init__(self, patch_size, num_patches, mask_coord=None):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_coord = mask_coord

    def __call__(self, img):
        _, w, h = img.size()
        left_upper, right_upper, right_lower, left_lower = self.mask_coord
        taken = list() # Rectangle: left_upper -> right_upper -> right_lower -> left_lower
        for _ in range(self.patch_size):
            flag = True
            i = 0
            while flag and taken:
                x0, y0 = torch.randint(low=16,high=w-15), torch.randint(low=16,high=h-15)
                x, y = taken[i]
                flag = math.abs(x, x0) > 16 and math.abs(y,y0) > 16 # NOTE: check whether it overlaps mask
                i += 1
            taken.append((x0,y0))

        patchs = []
        for x,y in taken:
            patch = img[:,x-16:x+16,y-16:y+16]
            patchs.append(patch)
            
        return patchs

class ConstancyNormalize:
    def __call__(self, img):
        return img
