import json
import time
import os
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from torchvision import transforms as tf
import numpy as np
import cv2
import wandb
from tqdm import tqdm

from data.comma_dataset import CommaDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.e2e_model import End2EndNet
from utils import paths, logging
from utils.losses import grad_l1_loss


def main(cfg):
    # Get wandb setup
    run = wandb.init(project="end-to-end-driving", config=cfg)
    
    # Create directory for saving checkpoints
    checkpoint_dir = Path(cfg['training']['checkpoint_dir']) / run.name
    checkpoint_dir.mkdir(parents=True)
    
    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizes training speed
    torch.backends.cudnn.benchmark = True
    
    print("Setting up datasets and data loaders...")
    # Define image transforms
    train_augs = tf.Compose([getattr(tf, name)(**kwargs) for name, kwargs in cfg['augs']['train'].items()])
    val_augs = tf.Compose([getattr(tf, name)(**kwargs) for name, kwargs in cfg['augs']['val'].items()])
    
    # Setup datasets and data loaders
    train_set = CommaDataset(cfg['dataset'], 'train', train_augs)
    train_loader = DataLoader(train_set, 
                              batch_size=cfg['train_loader']['batch_size'], 
                              shuffle=False, 
                              num_workers=cfg['train_loader']['num_workers'],
                              pin_memory=True)
    val_set = CommaDataset(cfg['dataset'], 'val', val_augs)
    val_loader = DataLoader(val_set, 
                              batch_size=cfg['val_loader']['batch_size'], 
                              shuffle=False, 
                              num_workers=cfg['val_loader']['num_workers'],
                              pin_memory=True)
    
    print("Initializing model and loss...")
    encoder = Encoder(cfg['model']['encoder'])
    decoder = Decoder(cfg['model']['decoder'])
    e2e_net = End2EndNet(encoder, decoder)
    e2e_net.to(device)

    # Init loss and optimizer
    l1_loss = torch.nn.L1Loss()
    optimizer = getattr(torch.optim, cfg['optimizer']['name'])(e2e_net.parameters(), **cfg['optimizer']['kwargs'])
    scheduler = getattr(torch.optim.lr_scheduler, cfg['scheduler']['name'])(optimizer, **cfg['scheduler']['kwargs'])
    
    # Create a GradScaler once at the beginning of training.
    scaler = amp.GradScaler()
    
    global_train_step = 0
    global_val_step = 0
    
    for epoch in range(cfg['training']['num_epochs']):
        # Training loop
        print(f"\n=== Epoch {epoch + 1} ===")
        running_train_loss = 0.0
        log_tensors = {"img": None, "label": None, "output": None}
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
            
            running_train_loss += loss.item()
            
            # Backward and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            
            # Track metrics
            if i_batch % cfg['training']['log_iterations']:
                # Track a random example from the batch
                idx = random.randrange(frames.shape[0])
                log_tensors['img'] = frames[idx][-1].detach().cpu().numpy()
                log_tensors['label'] = label[idx].detach().cpu().numpy()
                log_tensors['output'] = model_output[idx].detach().cpu().numpy()

                wandb.log({
                    'train_loss': loss,
                    'lr': scheduler.get_last_lr()[0],
                    'global_step': global_train_step
                })
                
            global_train_step += 1
            
            scheduler.step()

        # Generate log images and plots from last batch
        label_img, pred_img, fig = logging.gen_logs(log_tensors)
        
        # Log train epoch loss, images and plots
        wandb.log({
            'train_loss_epoch': running_train_loss / len(train_loader),
            'epoch': epoch,
            'train_labels': wandb.Image(label_img, caption=f"label"),
            'train_preds': wandb.Image(pred_img, caption=f"pred"),
            'train_3D_plots': wandb.Plotly(fig)
        })

        # Validation loop
        running_val_loss = 0.0
        e2e_net.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
                frames = sample_batched['frames'].to(device)
                label_path = sample_batched['label_path'].float().to(device)
                prev_path = sample_batched['prev_path'].float().to(device)
                
                # Combine last 2 dims in label to match model output
                label = label_path.view(label_path.shape[0], -1)

                # Forward pass
                model_output = e2e_net(frames, prev_path)
                loss = l1_loss(label, model_output)

                running_val_loss += loss.item()
                
                # Track metrics
                if i_batch % cfg['training']['log_iterations']:
                    # Track a random example from the batch
                    idx = random.randrange(frames.shape[0])
                    log_tensors['img'] = frames[idx][-1].detach().cpu().numpy()
                    log_tensors['label'] = label[idx].detach().cpu().numpy()
                    log_tensors['output'] = model_output[idx].detach().cpu().numpy()

                    wandb.log({'val_loss': loss, 'global_step': global_val_step})

                global_val_step += 1
            
        # Save model checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': e2e_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            },
            Path(checkpoint_dir) / f"checkpoint_{epoch}.tar"
        )
            
        # Generate log images and plots from last batch
        label_img, pred_img, fig = logging.gen_logs(log_tensors)

        # Log val epoch loss, LR, and log images/plots
        wandb.log(
            {
            'val_loss_epoch': running_val_loss / len(val_loader),
            'epoch': epoch,
            'val_labels': wandb.Image(label_img, caption=f"label"),
            'val_preds': wandb.Image(pred_img, caption=f"pred"),
            'val_3D_plots': wandb.Plotly(fig)
            }
        )


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
