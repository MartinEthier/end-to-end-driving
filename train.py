import json
import time
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import wandb

from data.comma_dataset import CommaDataset
from models.encoder import Encoder
from models.sequence_model import SequenceModel
from models.end_to_end_model import End2EndNet


#wandb.init(project="end-to-end-driving")


def main(cfg):
    # Define image transforms
    img_transforms = Compose([
        Resize([288, 384]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup datasets and data loaders
    train_set = CommaDataset(cfg['dataset'], 'train', img_transforms)
    train_loader = DataLoader(train_set, 
                              batch_size=cfg['data_loader']['batch_size'], 
                              shuffle=False, 
                              num_workers=cfg['data_loader']['num_workers'])
    val_set = CommaDataset(cfg['dataset'], 'val', img_transforms)
    val_loader = DataLoader(val_set, 
                              batch_size=cfg['data_loader']['batch_size'], 
                              shuffle=False, 
                              num_workers=cfg['data_loader']['num_workers'])
    
    # Initialize model
    encoder = Encoder(cfg['model']['encoder'])
    seq_model = SequenceModel(cfg['model']['sequence_model'])
    e2e_net = End2EndNet(encoder, seq_model)
    
    for epoch in range(cfg['training']['num_epochs']):
        for i_batch, sample_batched in enumerate(train_loader):
            frames = sample_batched['frames']
            label_path = sample_batched['label_path']
            prev_path = sample_batched['prev_path']
            print(frames.shape)
            print(label_path.shape)
            print(prev_path.shape)

            # Pass through e2e model
            model_output = e2e_net(frames)
            print(model_output.shape)
            break

        for i_batch, sample_batched in enumerate(val_loader):
            frames = sample_batched['frames']
            label_path = sample_batched['label_path']
            prev_path = sample_batched['prev_path']

            # Pass through e2e model
            model_output = e2e_net(frames)
            print(model_output.shape)
            break
        break


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_name', type=str, help='Name of the config file located in configs')
    args = parser.parse_args()
    
    # Load in config
    cfg_path = Path(__file__).parent.absolute() / 'configs' / args.cfg_name
    with cfg_path.open('r') as fr:
        cfg = json.load(fr)
    print(cfg)
    
    main(cfg)
