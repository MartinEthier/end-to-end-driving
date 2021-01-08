import json
import os
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.comma_dataset import CommaDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.e2e_model import End2EndNet


class CLR():
    """
    Yoinked from:
    https://medium.com/dsnet/the-1-cycle-policy-an-experiment-that-vanished-the-struggle-in-training-neural-nets-184417de23b9
    """
    def __init__(self, train_loader, base_lr=1e-5, max_lr=100):
        self.base_lr = base_lr # The lower boundary for learning rate (initial lr)
        self.max_lr = max_lr # The upper boundary for learning rate
        self.bn = len(train_loader) - 1 # Total number of iterations used for this test run (lr is calculated based on this)
        ratio = self.max_lr / self.base_lr # n
        self.mult = ratio ** (1 / self.bn) # q = (max_lr/init_lr)^(1/n)
        self.best_loss = 1e9 # our assumed best loss
        self.iteration = 0 # current iteration, initialized to 0
        self.lrs = []
        self.losses = []
        
    def calc_lr(self, loss):
        self.iteration += 1
        if math.isnan(loss) or loss > 4 * self.best_loss: # stopping criteria (if current loss > 4*best loss) 
            return -1
        if loss < self.best_loss and self.iteration > 1: # if current_loss < best_loss, replace best_loss with current_loss
            self.best_loss = loss
        mult = self.mult ** self.iteration # q = q^i
        lr = self.base_lr * mult # lr_i = init_lr * q
        self.lrs.append(lr) # append the learing rate to lrs
        self.losses.append(loss) # append the loss to losses
        return lr
        
    def plot(self, start=10, end=-5): # plot lrs vs losses
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log') # learning rates are in log scale
        plt.show()

def main(cfg):
    # Use gpu if available
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizes training speed
    torch.backends.cudnn.benchmark = True
    
    print("Setting up datasets and data loaders...")
    # Define image transforms
    img_transforms = Compose([
        Resize([288, 384]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup datasets and data loaders
    train_set = CommaDataset(cfg['dataset'], 'train', img_transforms)
    train_loader = DataLoader(train_set, 
                              batch_size=cfg['train_loader']['batch_size'], 
                              shuffle=False, 
                              num_workers=cfg['train_loader']['num_workers'],
                              pin_memory=True)
    
    print("Initializing model and loss...")
    # Init model and set multi-gpu if needed
    encoder = Encoder(cfg['model']['encoder'])
    decoder = Decoder(cfg['model']['decoder'])
    e2e_net = End2EndNet(encoder, decoder)
    e2e_net.to(device)
    
    # Init loss and optimizer
    l1_loss = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(e2e_net.parameters(), lr=1e-4, momentum=0.95)
    clr = CLR(train_loader)
    
    # Create a GradScaler once at the beginning of training.
    scaler = amp.GradScaler()
    
    # Training loop
    running_loss = 0.0
    avg_beta = 0.98 # useful in calculating smoothed loss
    e2e_net.train()
    for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
        optimizer.zero_grad()

        frames = sample_batched['frames'].to(device)
        label_path = sample_batched['label_path'].float().to(device)
        prev_path = sample_batched['prev_path'].float().to(device)

        # Combine last 2 dims in label to match model output
        label = label_path.view(label_path.shape[0], -1)

        # Run the forward pass with autocasting
        with amp.autocast():
            model_output = e2e_net(frames, prev_path) # (8, future*3)
            loss = l1_loss(label, model_output)
        
        # Calculate the smoothed loss (moving average)
        running_loss = avg_beta * running_loss + (1 - avg_beta) * loss.item()
        smoothed_loss = running_loss / (1 - avg_beta ** (i_batch + 1))
        
        # Calculate LR using CLR
        lr = clr.calc_lr(smoothed_loss)
        # Stopping criteria
        if lr == -1 :
            break
        # Update LR
        for pg in optimizer.param_groups:
            pg['lr'] = lr 

        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
        
    return clr


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_name', type=str, help='Name of the config file located in configs')
    args = parser.parse_args()
    
    # Load in config
    cfg_path = Path(__file__).parent.absolute() / args.cfg_name
    with cfg_path.open('r') as fr:
        cfg = json.load(fr)
    print(cfg)
    
    main(cfg)
