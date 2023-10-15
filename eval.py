'''
test.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 42th, 2023
CS 7180: Advnaced Perception

code for testing the network
'''
import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from model import SRCNN
import util

def main():
    '''
    Driver function to test the network
    '''
    # initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    args = parser.parse_args()

    # set up device and initialize the network
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    state_dict = model.state_dict()

    # load the saved parameters
    for n, p in torch.load(args.weights_file, map_location= lambda storage, loc: storage).items():
        if n in state_dict.keys(): state_dict[n].copy_(p)
        else: raise KeyError(n)
    model.eval()

    # populate bicubic image of original
    image = pil_image.open(args.image_file).convert('RGB')
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
    image = np.array(image).astype(np.float32)

    # reconstructiong to fit the network
    ycbcr = convert_rgb_to_ycbcr(image)
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    # input to the network
    with torch.no_grad():
        preds = model(y).clamp(0., 1.)
    
    psnr = utils.psnr(y, preds)
    print('psnr: {:.2f}'.format(psnr))

    # save the output of SRCNN
    preds = preds.mul(255.).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1,2,0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0., 255.).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))

if __name__ == '__main__':
    main()