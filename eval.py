'''
eval.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 16th, 2023
CS 7180: Advnaced Perception

Evaluate the CCCNN
'''
import os
import argparse
import numpy as np
import PIL.Image as pil_image
import cv2
import matplotlib.pyplot as plt

# torch
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# custom
from model import CCCNN
from dataset import CustomDataset, ReferenceDataset
from util import angularLoss, illuminate

def main():
    '''
    Driver function to test the network
    '''
    # initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-space', default=False, action='store_true')
    parser.add_argument('--num-patches', type=int, required=True)
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
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
    ref_dataset = ReferenceDataset(args.images_dir, args.labels_file)
    eval_dataset = CustomDataset(args.images_dir, args.labels_file, log_space=args.log_space, num_patches=args.num_patches)
    eval_dataloader = DataLoader(dataset=eval_dataset, 
                                batch_size=1,
                                num_workers=args.num_workers
                                )
    
    losses = []
    for idx, (batch, data) in enumerate(zip(eval_dataloader, ref_dataset)):
        input, label = data
        inputs, labels = batch
        inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, ...] -> [batch size * num_patches, ...] / NOTE: optimize?
        labels = torch.flatten(labels, start_dim=0, end_dim=1)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad(): preds = model(inputs)

        mean_pred = torch.mean(preds, dim=0)
        loss = angularLoss(mean_pred, label) / preds.shape[0]
        losses.append(loss)

        label_img = illuminate(input, label)
        pred_img = illuminate(input, mean_pred)
        cv2.imwrite(os.path.join(args.outputs_dir,'label_{}'.format(idx)), label_img)
        cv2.imwrite(os.path.join(args.outputs_dir,'pred_{}'.format(idx)), pred_img)

    # calculate stats
    losses.sort()
    l = len(losses)
    minimum = min(losses)
    tenth = losses[l//10]
    median = losses[l//2]
    average = sum(losses) / l
    ninetieth = losses[l * 9 // 10]
    maximum = max(losses)

    print("Min: {}\n10th per: {}\nMed: {}\nAvg: {}\n 90th per: {}\nMax: {}\n".format(minimum, tenth, median, average, ninetieth, maximum))

    # draw histogram
    plt.hist(losses)


if __name__ == '__main__':
    main()