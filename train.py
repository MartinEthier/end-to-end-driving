import json
import time
import os
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
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
    #gpu_str = ",".join([str(gid) for gid in cfg['training']['gpu_ids']])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    val_set = CommaDataset(cfg['dataset'], 'val', img_transforms)
    val_loader = DataLoader(val_set, 
                              batch_size=cfg['val_loader']['batch_size'], 
                              shuffle=False, 
                              num_workers=cfg['val_loader']['num_workers'],
                              pin_memory=True)
    
    print("Initializing model and loss...")
    # Init model and set multi-gpu if needed
    encoder = Encoder(cfg['model']['encoder'])
    decoder = Decoder(cfg['model']['decoder'])
    e2e_net = End2EndNet(encoder, decoder)
    if torch.cuda.device_count() > 1:
        print(f"Using multiple GPUs")
        e2e_net = torch.nn.DataParallel(e2e_net)
    e2e_net.to(device)
    wandb.watch(e2e_net)

    # Init loss and optimizer
    l1_loss = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(e2e_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        2e-1,
        epochs=cfg['training']['num_epochs'],
        steps_per_epoch=len(train_loader)
    )
    
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
