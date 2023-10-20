# source: https://github.comVisillect/CubePlusPlus/blob/master/challenge/make_preview.py
import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from util import read_16bit_png, RandomPatches
import pandas as pd

cam2rgb = np.array([
    1.8795, -1.0326, 0.1531,
    -0.2198, 1.7153, -0.4955,
    0.0069, -0.5150, 1.5081,]).reshape((3, 3))

def main():
    download_dir = "./fig/"
    idx = 5
    
    images_dir = Path("./SimpleCube++/test/PNG/")
    gt_dir = Path("./SimpleCube++/auxiliary/JPG/")
    lables = pd.read_csv("./SimpleCube++/test/gt.csv")
    images = os.listdir(images_dir)
    gts = os.listdir(gt_dir)

    image_path = os.path.join(images_dir,images[idx])
    gt_path = os.path.join(gt_dir, gts[idx])

    print(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gt = np.float32(cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)/255)
    labelR, labelG, labelB = lables.iloc[idx][1:4] # RGB
    
    # print(image) # BGR
    # print(image.shape) -> (row, column): (432,648,3), width x height: 648x432
    # right bottom rectangle width x height: 175x250 -> (232, 473)
    # it looks like it is actually 175x250

    cv2.imshow('cv2', image)
    # cv2.imwrite("PNG", image)
    cv2.waitKey(0)

    # # image = read_16bit_png(image_path).permute(1,2,0)
    # # print("by torch:")
    # # print(image) # RGB
    # # print(image.shape)
    # # cv2.imshow('torch', image.tolist())
    # # cv2.waitKey(0)

    linearized_img = linearize(image)
    cv2.imshow('linearized img', linearized_img)
    # cv2.imwrite("linearized_PNG",linearized_img)
    cv2.waitKey(0)

    sRGB_img = L2sRGB(linearized_img)
    cv2.imshow('sRGB img', sRGB_img)
    cv2.waitKey(0)
    ### Bruce's visualization: x = 255 * x / 11.4

    # eps = 1e-7
    # log_sRGB_8bit_img = np.log(sRGB_img * 255) / 5.54
    # cv2.imshow('log sRGB 8bit img', log_sRGB_8bit_img)
    # cv2.waitKey(0)

    # log_sRGB_16bit_img = np.log(sRGB_img * 65535) / 11.4
    # cv2.imshow('log sRGB 16bit img', log_sRGB_16bit_img)
    # cv2.waitKey(0)

    cv2.imshow('gt', gt)
    cv2.waitKey(0)

    preview = get_preview_csv(image, idx)
    cv2.imshow('preview', preview)
    cv2.waitKey(0)



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

def get_preview_csv(img, idx, field_name="gt"):
    labels = pd.read_csv("./SimpleCube++/test/gt.csv")
    illum = labels.iloc[idx, 1:4].astype(float).values # rgb, e.g. .2, .4, .4
    print(illum)
    
    # linearized_img = linearize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64))
    linearized_img = linearize(img)
    cv2.imshow('linearized_img', linearized_img)
    cv2.waitKey(0)

    white_balanced_image = np.clip(linearized_img/illum, 0, 1) # lin ing
    cv2.imshow('white_balanced_image', white_balanced_image)
    cv2.waitKey(0)

    rgb_img = np.dot(white_balanced_image, cam2rgb.T)
    cv2.imshow('rgb_img', rgb_img)
    cv2.waitKey(0)
    rgb_img = np.clip(rgb_img, 0, 1)**(1/2.2)
    return (rgb_img*255).astype(np.uint8)

def get_preview(img_png_path, field_name="original_gt"):
    with open("./Cube++/auxiliary/source/JPG.JSON/00_0057.jpg.json") as meta:
        illum = np.array(json.load(meta)[field_name])
        print("illum: ")
        print(illum)
        illum /= illum.sum()
        print("normalized illum: ")
        print(illum)

    
    cam = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)
    cv2.imshow('cam', cam)
    cv2.waitKey(0)

    # cam = linearize(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB).astype(np.float64))
    cv2.imshow('linearized_cam', cam)
    cv2.waitKey(0)

    cam_wb = np.clip(cam/illum, 0, 1)
    cv2.imshow('white balanced cam', cam)
    cv2.waitKey(0)

    rgb = np.dot(cam_wb, cam2rgb.T)
    cv2.imshow('rgb', rgb)
    cv2.waitKey(0)

    rgb = np.clip(rgb, 0, 1)**(1/2.2)
    cv2.imshow('gamma corrected rgb', rgb)
    cv2.waitKey(0)
    return (rgb*255).astype(np.uint8)


if __name__ == "__main__":
    main()
