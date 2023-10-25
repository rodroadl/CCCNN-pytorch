'''
train.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 20th, 2023
CS 7180: Advnaced Perception

code for training the network
'''

import argparse
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

# custom
from model import CCCNN
from dataset import CustomDataset
from util import angularLoss

def main():
    '''
    Driver function to train the network
    '''
    # setting up argumentparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-space', type=str, default='linear')
    parser.add_argument('--label-space', type=str, default='linear')
    parser.add_argument('--num-patches', type=int, required=True)
    parser.add_argument('--train-images-dir', type=str, required=True)
    parser.add_argument('--train-labels-file', type=str, required=True)
    parser.add_argument('--eval-images-dir', type=str, required=True)
    parser.add_argument('--eval-labels-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    # args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir): os.makedirs(args.outputs_dir)

    # set up device, instantiate the SRCNN model, set up criterion and optimizer
    cudnn.benchmark = True
    # cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model = CCCNN().to(device)
    criterion = nn.MSELoss(reduction="sum") # NOTE: check, euclidean loss?
    optimizer = optim.Adam([
        {'params': model.conv.parameters()},
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()}
    ], lr=args.lr)

    # (Initialize logging)
    print(f'''Starting training:
        Image Space:    {args.image_space}
        Label Space:    {args.label_space}
        Epoch:          {args.num_epochs}
        Batch size:     {args.batch_size}
        Learning rate:  {args.lr}
        Device:         {device.type}
    ''')

    # configure datasets and dataloaders
    train_dataset = CustomDataset(args.train_images_dir, args.train_labels_file, num_patches=args.num_patches, image_space=args.image_space, label_space=args.label_space)
    eval_dataset = CustomDataset(args.eval_images_dir, args.eval_labels_file, num_patches=args.num_patches, image_space=args.image_space, label_space=args.label_space)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, 
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers
                                 )

    # track best parameters and values
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float('inf')
    train_loss_log = list()
    eval_loss_log = list()

    # start the training
    
    for epoch in range(args.num_epochs):
        model.train()

        with tqdm(total=(len(train_dataset)- len(train_dataset)% args.batch_size)) as train_pbar:
            train_pbar.set_description('train epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for batch in train_dataloader:
                inputs, labels = batch
                inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, ...] -> [batch size * num_patches, ...] / NOTE: optimize?
                labels = torch.flatten(labels, start_dim=0, end_dim=1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                if torch.isnan(inputs).any():
                    print("nan inputs found")
                    raise SystemExit
                preds = model(inputs)
                if torch.isnan(preds).any():
                    print("nan preds found")
                    raise SystemExit
                loss = criterion(preds,labels)
                train_loss_log.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                for _, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print("nan gradient found")
                        raise SystemExit
                optimizer.step()
                train_pbar.update(args.batch_size)

        with tqdm(total=(len(eval_dataset))) as eval_pbar:
            eval_pbar.set_description('eval round:')
            # start the evaluation
            model.eval()
            round_loss = 0
            num_patches = 0
            for batch in eval_dataloader:
                inputs, labels = batch
                inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, ...] -> [batch size * num_patches, ...] / NOTE: optimize?
                labels = torch.flatten(labels, start_dim=0, end_dim=1)
                num_patches += inputs.shape[0]
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad(): preds = model(inputs)
                batch_loss = angularLoss(preds, labels)
                round_loss += batch_loss
                eval_pbar.update(args.batch_size)
            round_loss /= num_patches
            eval_loss_log.append(round_loss)
            print('eval round loss: {:.2f}'.format(round_loss))

            # update best parameters and values
            if best_loss > round_loss:
                best_epoch = epoch
                best_loss = round_loss
                best_weights = copy.deepcopy(model.state_dict())
    print('best epoch: {}, angular loss: {:.2f}'.format(best_epoch, best_loss))
    torch.save(best_weights, os.path.join(args.outputs_dir, '{}2{}_lr{}_{}.pth'.format(args.image_space, args.label_space,args.lr, best_loss)))

    plt.figure()
    plt.subplot(211)
    plt.plot(range(len(train_loss_log)), train_loss_log)
    plt.subplot(212)
    plt.plot(range(len(eval_loss_log)), eval_loss_log)
    plt.show()

if __name__ == "__main__":
    main()
    