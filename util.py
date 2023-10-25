'''
util.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 24th, 2023
CS 7180: Advnaced Perception

File containing utility functions

variable:
    cam2rgb - Global variable, transformation matrix for cam to RGB

function:
    read_16bit_png - read 16bit png file using torch
    angularLoss - calculate accumulated angular loss in degrees
    illuminate - Linearize, illuminate, map to RGB and gamma correct
    L2sRGB - Map linear chromaticity space to sRGB chromaticity space
    to_rgb - Map input to rgb chromaticity space

class:
    MaxResize - scale input resizing longer size to max
    ContrastNormalization - apply histogram stretching to normalize contrast
    RandomPatches - randomly crop image to number of 32x32 patches
'''
# built-in
import math 
from random import sample

# third-party
import numpy as np

# torch
import torch
from torchvision.io import read_file
from torchvision.transforms import functional as F

cam2rgb = np.array([
    1.8795, -1.0326, 0.1531,
    -0.2198, 1.7153, -0.4955,
    0.0069, -0.5150, 1.5081,]).reshape((3, 3))

def read_16bit_png(file):
    '''
    Return 16 bit image

    Parameter:
        file(str or Path) - 16bit image file

    Return:
        16bit image
    '''
    data = read_file(file)
    return torch.ops.image.decode_png(data, 0, True)

def angularLoss(xs, ys):
    '''
    Return accumulated angular loss in degrees

    Parameter:
        xs(tensor or sequence of tensors) - sequence of tensors to calculate angular loss
        ys(tensor or sequence of tensors) - sequence of tensors to calculate angular loss

    Return:
        output(float) - accumulated angular loss in degrees
    '''
    if xs.shape[0] == 1:
        print("angular loss - xs[0], ys[0]:", xs[0], ys[0])
        if torch.count_nonzero(xs[0]).item() == 0: return 180
        return math.degrees(torch.arccos(torch.nn.functional.cosine_similarity(xs,ys, dim=-1)).item())
    
    output = 0
    for x, y in zip(xs, ys):
        if torch.count_nonzero(x).item() == 0: output += 180
        else: output += math.degrees(torch.arccos(torch.nn.functional.cosine_similarity(x,y, dim=0)).item())
    return output

def illuminate(img, illum):
    '''
    Linearize, illuminate, map to RGB and gamma correct

    Parameter:
        img(tensor) - image to process
        idx(int) - index to get illumination

    Return:
        output(numpy.ndarray) - 
    '''
    linearize = ContrastNormalization()
    linearized_img = linearize(img).permute(1,2,0).cpu().numpy() # h,w,c -> c,h,w
    illum = illum.cpu().numpy()
    white_balanced_image = linearized_img/illum
    rgb_img = np.dot(white_balanced_image, cam2rgb.T)
    rgb_img = np.clip(rgb_img, 0, 1)**(1/2.2)
    return (rgb_img*255).astype(np.uint8)

def L2sRGB(linImg):
    '''
    Map linear chromaticity space to sRGB chromaticity space

    Parameter:
        linImg(tensor) - image in linear chromaticity space

    Return:
        linImg(tensor) - image map to sRGB chromaticity space
    '''
    low_mask = linImg <= 0.0031308
    high_mask = linImg > 0.0031308
    linImg[low_mask] *= 12.92
    linImg[high_mask] = 1.055 * linImg[high_mask]**(1/2.4) - 0.055
    return linImg

def to_rgb(inputs):
    '''
    Map input to rgb chromaticity space (r,g,b in [0,1] such that r+g+b = 1)

    Parameter:
        input(tensor) - num_patches x 3, input in arbitrary chromaticity space

    Return:
        input(tensor) - input mapped to rgb chromaticity space
    '''
    num_patches = inputs.shape[0]
    if num_patches == 1: return inputs[0] / torch.sum(inputs[0])
    for idx in range(num_patches):
        print("before norm:", inputs[idx])
        inputs[idx] = inputs[idx] / torch.sum(inputs[idx])
        print("after norm:", inputs[idx])
    return inputs

#################
### Transform ### 
#################
class MaxResize:
    '''
    Downscale input while longer side is capped at self.max_length
    '''
    def __init__(self, max_length):
        '''
        Constructor

        Parameters:
            max_length(int) - maximum length to downscale
        '''
        self.max_length = max_length
        
    def __call__(self, img):
        '''
        Return downscaled image while longer side is capped at self.max_length

        Parameter:
            img(tensor) - image to downscale
        Return:
            downscaled image wheer longer side is capped at self.max_length
        '''
        _, h, w = img.size()
        ratio = float(h) / float(w)
        if ratio > 1: # h > w
            w0 = math.ceil(self.max_length / ratio)
            return F.resize(img, (self.max_length, w0), antialias=True)
        else: # h <= w
            h0 = math.ceil(self.max_length / ratio)
            return F.resize(img, (h0, self.max_length), antialias=True)

class ContrastNormalization:
    '''
    Apply Global Histogram Stretching to normalize contrast
    '''
    def __init__(self, black_lvl=2048):
        '''
        Constructor

        Parameter:
            black_lvl(int, optional) - value of black orginall captured by camera
        '''
        self.black_lvl = black_lvl
    def __call__(self, img):
        '''
        Return contrast normalized image

        Parameters:
            img(tensor) - image to contrast normalize

        Return:
            output(tensor) - contrast normalized image in [0,1]
        '''
        saturation_lvl = torch.max(img)
        return torch.clamp((img - self.black_lvl)/(saturation_lvl - self.black_lvl),0,1)

class RandomPatches:
    '''
    Randomly crop image to number of 32x32 patches
    '''
    def __init__(self, patch_size, num_patches):
        '''
        Constructor

        Parameters:
            patch_size(int) - size of patch
            num_patches(int) - number of patch to return
        '''
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, img):
        '''
        Return sequences of 32x32 patches that was randomly cropped from image

        Parameter:
            img(tensor) - image to radomly crop 32x32 patches
        Return:
            sequences of 32x32 patches that was randomly cropped from image
        '''
        # specific to SimpleCube++
        MASK_HEIGHT = 250
        MASK_WIDTH = 175

        # assign and initiate variables
        _, h, w = img.size()        
        diameter = self.patch_size
        radius = self.patch_size // 2
        coords = set()
        center = list()

        # populate candidate for center of patches
        for row in range(radius, h-radius):
            for col in range(radius, w-radius):
                if (row < h-radius-MASK_HEIGHT or col < w-radius-MASK_WIDTH): coords.add((row, col)) # check whether it overlap with masked rectangle

        # sample center for patches
        for _ in range(self.num_patches):
            valid = False
            while coords and not valid:
                y0, x0 = sample(coords, 1)[0]
                coords.remove((y0, x0))
                valid = True
                for y, x in center:
                    if not valid: break # if overlap, try another one
                    valid &= abs(y-y0) > diameter and abs(x-x0) > diameter # check whether it overlap with other patches
            if valid: center.append((y0,x0)) # if it doesn't overlap, sample it

        # sample patches according to chosen centers
        patches = []
        for y,x in center:
            patch = img[:, y-16:y+16, x-16:x+16].type(torch.float32)
            patches.append(patch)

        return torch.stack(patches, dim=0) # list of tensors -> sequence(tensor) of tensors