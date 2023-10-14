# source: https://github.comVisillect/CubePlusPlus/blob/master/challenge/make_preview.py
import os
import cv2
import numpy as np
from pathlib import Path

def main():
    root = Path("./SimpleCube++/train/PNG/")
    images = os.listdir(root)
    idx = 5
    image = cv2.imread(os.path.join(root,images[idx]), cv2.IMREAD_UNCHANGED)
    image = linearize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64))

    cv2.imshow('window', image)
    cv2.waitKey(0)



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
