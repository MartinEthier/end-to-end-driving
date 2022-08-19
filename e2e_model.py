import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision import transforms as tf
from torchvision import utils as tv_utils
import yaml
import wandb
import numpy as np

from data.comma_dataset import CommaDataset
import data.transforms as dtf
from models.encoder import Encoder
from models.decoder import Decoder
import utils.logging as lg


class CommaDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        train_img_augs = tf.Compose([getattr(tf, name)(**kwargs) for name, kwargs in self.cfg['train_augs']['img_augs'].items()])
        train_full_augs = tf.Compose([getattr(dtf, name)(**kwargs) for name, kwargs in self.cfg['train_augs']['full_augs'].items()])
        val_augs = tf.Compose([getattr(tf, name)(**kwargs) for name, kwargs in self.cfg['val_augs'].items()])

        self.train_set = CommaDataset(self.cfg, 'train', train_img_augs, train_full_augs)
        self.val_set = CommaDataset(self.cfg, 'val', val_augs, None)
        
        # Calculate sample weights for WeightedRandomSampler
        num_bins = self.cfg['dataset']['num_bins']
        w = self.cfg['dataset']['weighing_factor']
        curv = self.train_set.curvatures
        counts, bins = np.histogram(curv, bins=num_bins, density=True)
        # Replace 0 counts with 1 to avoid divide-by-zero warnings
        counts[counts == 0] = 1
        bin_weights = 1 / counts**w
        sample_bin_idx = np.digitize(curv, bins, right=False) - 1
        # Largest example has idx == num_bins because of right=False, just use previous weight
        sample_bin_idx[sample_bin_idx == num_bins] = num_bins - 1
        sample_weights = bin_weights[sample_bin_idx]
        self.sampler = torch.utils.data.WeightedRandomSampler(sample_weights, self.cfg['training']['num_steps'])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["num_workers"],
            sampler=self.sampler,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["num_workers"],
            shuffle=True,
            pin_memory=True
        )

class LitE2EModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg, self.device)

    def forward(self, frames, prev_path):
        """
        frames shape: (batch_size, seq_len, 3, height, width)
        prev_path shape: (batch_size, seq_len, 3)
        """
        # Combine frames along channel dim
        if self.cfg['dataset']['channel_concat']:
            s = frames.shape
            frames = frames.view(s[0], 1, s[1]*s[2], s[3], s[4])

        # Pass sequence through the encoder
        image_features = []
        for t in range(frames.shape[1]):
            image_features.append(self.encoder(frames[:, t])) # (batch_size, enc_feat_len)
            
        # Combine image features into a sequence
        feature_sequence = torch.stack(image_features) # (seq_len, batch_size, enc_feat_len)

        # Pass through decoder
        if self.cfg['training']['seq_model_decoder']:
            # Init x0 to a zero vec same size as yt
            # Init h0 to encoder output
            output = torch.zeros((1, frames.shape[0], 3), device=self.device)
            hidden = feature_sequence
            outputs = []
            for i in range(self.cfg['dataset']['future_steps']):
                output, hidden = self.decoder(output, hidden)
                outputs.append(output)
            # Reshape to (B, 90)
            # list of len 30 of (1, batch_size, 3)
            model_output = torch.cat(outputs, dim=0).transpose(0, 1)
            # (batch_size, 30, 3)
        else:
            model_output = self.decoder(feature_sequence).view(frames.shape[0], -1, 3) # (batch_size, future_steps, 3)
        
        return model_output

    def training_step(self, sample_batched, batch_idx):
        frames = sample_batched['frames']
        label_path = sample_batched['label_path']
        prev_path = sample_batched['prev_path']

        model_output = self(frames, prev_path) # (B, future_steps, 3)
        loss = F.l1_loss(label_path, model_output)

        self.log('train_loss', loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, sample_batched, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_loss', summary='min')

        frames = sample_batched['frames']
        label_path = sample_batched['label_path']
        prev_path = sample_batched['prev_path']

        model_output = self(frames, prev_path) # (B, future_steps, 3)
        loss = F.l1_loss(label_path, model_output)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
        # Log images of the predictions
        if batch_idx == 0:
            num_imgs = self.cfg['training']['num_log_imgs']
            img_list = lg.log_viz(frames[:num_imgs].detach().cpu(), label_path[:num_imgs].detach().cpu(), model_output[:num_imgs].detach().cpu())
            self.logger.log_image(key='val_viz', images=img_list)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg['optimizer']['name'])(self.parameters(), **self.cfg['optimizer']['kwargs'])
        scheduler = getattr(torch.optim.lr_scheduler, self.cfg['scheduler']['name'])(optimizer, **self.cfg['scheduler']['kwargs'])
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        # Lowers memory use and improves performance
        optimizer.zero_grad(set_to_none=True)

def main():
    pl.seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_name', type=str, help='Name of the config file located in configs')
    parser.add_argument('--find_lr', '-f', action='store_true', help='Run LR finder test instead of training')
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.absolute() / args.cfg_name
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    
    # Prep logger and callbacks
    wandb_logger = pl.loggers.WandbLogger(project="end-to-end-driving", entity="methier")
    wandb_logger.experiment.config.update(cfg)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=True)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        dirpath=Path(cfg["training"]["checkpoint_dir"]) / wandb.run.name,
        filename="best-step_{step}-val_loss_{val_loss:.2f}",
        auto_insert_metric_name=False
    )

    # Create trainer
    dm = CommaDataModule(cfg)
    model = LitE2EModel(cfg)
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=16,
        amp_backend="native",
        benchmark=True,
        max_steps=cfg["training"]["num_steps"],
        log_every_n_steps=cfg["training"]["log_steps"],
        val_check_interval=cfg["training"]["val_interval"],
        logger=wandb_logger,
        auto_lr_find=args.find_lr,
        callbacks=[lr_monitor, model_checkpoint]
    )
    if args.find_lr:
        trainer.tune(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
