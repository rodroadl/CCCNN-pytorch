# source: https://github.comVisillect/CubePlusPlus/blob/master/challenge/make_preview.py
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from util import read_16bit_png
import pandas as pd

def main():
    idx = 4
    
    images_dir = Path("./SimpleCube++/test/PNG/")
    gt_dir = Path("./SimpleCube++/auxiliary/JPG/")
    lables = pd.read_csv("./SimpleCube++/test/gt.csv")
    images = os.listdir(images_dir)
    gts = os.listdir(gt_dir)

    image_path = os.path.join(images_dir,images[idx])
    gt_path = os.path.join(gt_dir, gts[idx])

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gt = np.float32(cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)/255)
    labelR, labelG, labelB = lables.iloc[idx][1:4] # RGB
    
    print("by cv2:") 
    # print(image) # BGR
    # print(image.shape)
    cv2.imshow('cv2', image)
    cv2.waitKey(0)

    # image = read_16bit_png(image_path).permute(1,2,0)
    # print("by torch:")
    # print(image) # RGB
    # print(image.shape)
    # cv2.imshow('torch', image.tolist())
    # cv2.waitKey(0)

    linearized_img = linearize(image)
    cv2.imshow('linearized img', linearized_img)
    cv2.waitKey(0)

    sRGB_img = L2sRGB(linearized_img)
    cv2.imshow('sRGB img', sRGB_img)
    cv2.waitKey(0)

    cv2.imshow('gt', gt)
    cv2.waitKey(0)

    gt[:,:,0] -= labelB
    gt[:,:,1] -= labelG
    gt[:,:,2] -= labelR
    cv2.imshow('gt without illumination', gt)
    cv2.waitKey(0)

def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    '''
    parameters:
        img - image
        black_lvl - value of black color when captured from camera
        saturation_lvl = maximum value
    '''
    return np.clip((img - black_lvl)/(saturation_lvl-black_lvl), 0, 1)

def L2sRGB(linImg):
    low_mask = linImg <= 0.0031308
    high_mask = linImg > 0.0031308
    linImg[low_mask] *= 12.92
    linImg[high_mask] = 1.055 * linImg[high_mask]**(1/2.4) - 0.055
    return linImg


if __name__ == "__main__":
    main()
