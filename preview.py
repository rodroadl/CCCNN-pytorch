# source: https://github.comVisillect/CubePlusPlus/blob/master/challenge/make_preview.py
import os
import cv2
import numpy as np
from pathlib import Path
from util import read_16bit_png

def main():
    root = Path("./SimpleCube++/train/PNG/")
    images = os.listdir(root)
    idx = 5
    image_path = os.path.join(root,images[idx])
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    print("by cv2:") 
    print(image) # BGR
    print(image.shape)

    image = read_16bit_png(image_path).permute(1,2,0)
    print("by torch:")
    print(image) # RGB
    print(image.shape)



def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    '''
    parameters:
        img - image
        black_lvl - value of black color when captured from camera
        saturation_lvl = maximum value
    '''
    return np.clip((img - black_lvl)/(saturation_lvl-black_lvl), 0, 1)

if __name__ == "__main__":
    main()
