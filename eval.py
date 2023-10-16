'''
eval.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 16th, 2023
CS 7180: Advnaced Perception

Evaluate the CCCNN
'''

import argparse
import numpy as np
import PIL.Image as pil_image

# torch
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# custom
from model import CCCNN
from dataset import CustomDataset
from util import angularLoss

def main():
    '''
    Driver function to test the network
    '''
    # initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    args = parser.parse_args()

    # set up device and initialize the network
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CCCNN().to(device)
    state_dict = model.state_dict()

    # load the saved parameters
    for n, p in torch.load(args.weights_file, map_location= lambda storage, loc: storage).items():
        if n in state_dict.keys(): state_dict[n].copy_(p)
        else: raise KeyError(n)
    model.eval()

    # configure datasets and dataloaders
    eval_dataset = CustomDataset(args.eval_file, "./SimpleCube++/test/gt.csv", log_space=args.log_space)
    eval_dataloader = DataLoader(dataset=eval_dataset, 
                                batch_size=1,
                                num_workers=args.num_workers
                                )
    
    for batch in eval_dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad(): preds = model(inputs)
        batch_loss = angularLoss(preds, labels)

    # save the output of SRCNN
    preds = preds.mul(255.).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1,2,0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0., 255.).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))

if __name__ == '__main__':
    main()