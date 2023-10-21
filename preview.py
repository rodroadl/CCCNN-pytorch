'''
preview.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 20th, 2023
CS 7180: Advnaced Perception

script to preview images
'''
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch


from util import read_16bit_png, L2sRGB, ContrastNormalization, illuminate


def main():
    '''
    Driver
    '''
    download_dir = "./fig/"
    idx = 5
    linearize = ContrastNormalization()
    
    images_dir = Path("./SimpleCube++/test/PNG/")
    gt_dir = Path("./SimpleCube++/auxiliary/JPG/")
    labels = pd.read_csv("./SimpleCube++/test/gt.csv")
    images = os.listdir(images_dir)
    illum = torch.tensor(labels.iloc[idx, 1:4].astype(float).values, dtype=torch.float32) 
    gts = os.listdir(gt_dir)

    image_path = os.path.join(images_dir,images[idx])
    gt_path = os.path.join(gt_dir, gts[idx])

    gt = np.float32(cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)/255)
    
    t = read_16bit_png(image_path) # RGB
    cv2.imshow('torch', t.permute(1,2,0)[:,:,[2,1,0]].numpy().astype(np.uint16))
    cv2.waitKey(0)

    illum_t = illuminate(t, illum)
    cv2.imshow('illum_t', illum_t)
    cv2.imwrite('illum_t.jpg', illum_t)
    cv2.waitKey(0)

    # linearized_img = linearize(image)
    # cv2.imshow('linearized img', linearized_img)
    # cv2.imwrite("linearized_PNG",linearized_img)
    # cv2.waitKey(0)

    # sRGB_img = L2sRGB(linearized_img)
    # cv2.imshow('sRGB img', sRGB_img)
    # cv2.waitKey(0)
    ### Bruce's visualization: x = 255 * x / 11.4

    # eps = 1e-7
    # log_sRGB_8bit_img = np.log(sRGB_img * 255) / 5.54
    # cv2.imshow('log sRGB 8bit img', log_sRGB_8bit_img)
    # cv2.waitKey(0)

    # log_sRGB_16bit_img = np.log(sRGB_img * 65535) / 11.4
    # cv2.imshow('log sRGB 16bit img', log_sRGB_16bit_img)
    # cv2.waitKey(0)

    # cv2.imshow('gt', gt)
    # cv2.waitKey(0)

    # illum_img = illuminate(image, idx)
    # cv2.imshow('preview', img)
    # cv2.waitKey(0)



    # rp = RandomPatches(32, 3)
    # patches = rp(gt)
    # for idx, patch in enumerate(patches):
    #     cv2.imshow('patch{}'.format(idx), patch)
    #     cv2.waitKey(0)

    # gt[gt != 0] = np.log(gt[gt != 0] * 255) / 5.54
    # cv2.imshow('log gt img', gt)
    # cv2.waitKey(0)

    # gt[:,:,0] -= labelB
    # gt[:,:,1] -= labelG
    # gt[:,:,2] -= labelR
    # cv2.imshow('gt without illumination', gt)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
