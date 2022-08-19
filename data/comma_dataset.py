from pathlib import Path
import json

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import io
from torchvision.transforms import ToTensor

from utils import paths, logging


class CommaDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, cfg, split, img_transforms, full_transforms):
        """
        cfg:
        """
        self.return_curvature = False
        self.cfg = cfg['dataset']
        self.img_transforms = img_transforms
        self.full_transforms = full_transforms
        
        self.to_tensor = ToTensor()
        
        # Load in trainval json
        json_path = Path(__file__).parent / 'dataset_lists' / self.cfg['dataset_file']
        with json_path.open('r') as fr:
            trainval_dict = json.load(fr)
        self.args = trainval_dict['args']
        
        # Set files and dataset size
        if split == 'train':
            self.dataset_size = self.args['train_size']
            self.frame_paths = [Path(self.args['root_dir']) / p[0] for p in trainval_dict['train_files']]
            self.curvatures = [p[1] for p in trainval_dict['train_files']]
        elif split == 'val':
            self.dataset_size = self.args['val_size']
            self.frame_paths = [Path(self.args['root_dir']) / p[0] for p in trainval_dict['val_files']]
            self.curvatures = None
        else:
            raise ValueError("Invalid split string (must be train or val):", self.split)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Get paths and frame id
        img_path = self.frame_paths[idx]
        route_path = img_path.parent.parent
        frame_id = int(img_path.stem)
        
        # Load route data arrays
        orientations = np.load(route_path / "frame_orientations.npy")
        positions = np.load(route_path / "frame_positions.npy")
        
        # Convert positions to reference frame
        local_path = paths.get_local_path(positions, orientations, frame_id)
        
        # Divide data into previous and future arrays
        future_path = local_path[frame_id + 1 : frame_id + 1 + self.cfg['future_steps']]
        previous_path = local_path[frame_id - self.cfg['past_steps'] : frame_id + 1]
        
        # Grab previous and current frames
        frames = []
        for f_id in range(frame_id - self.cfg['past_steps'], frame_id + 1):
            filename = str(f_id).zfill(6) + '.jpg'
            frame = Image.open(str(route_path / 'images' / filename))
            frames.append(self.to_tensor(frame))
        
        # Stack frames into single array, (T, C, H, W)
        frames = torch.stack(frames)
        
        if self.img_transforms is not None:
            frames = self.img_transforms(frames)

        sample = {}
        sample['frames'] = frames
        sample['label_path'] = torch.from_numpy(future_path).float()
        sample['prev_path'] = torch.from_numpy(previous_path).float()
        
        if self.full_transforms is not None:
            sample = self.full_transforms(sample)
        
        if self.return_curvature:
            sample['curv'] = self.curvatures[idx]

        return sample


if __name__=="__main__":
    from torchvision import transforms as tf
    import yaml
    import data.transforms as dtf
    from matplotlib import pyplot as plt

    # Load config file to test dataset class and augs
    cfg_name = "configs/resnet34_noseq.yaml"
    cfg_path = Path(__file__).parent.parent.absolute() / cfg_name
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    train_img_augs = tf.Compose([getattr(tf, name)(**kwargs) for name, kwargs in cfg['train_augs']['img_augs'].items()])
    train_full_augs = tf.Compose([getattr(dtf, name)(**kwargs) for name, kwargs in cfg['train_augs']['full_augs'].items()])
    val_augs = tf.Compose([getattr(tf, name)(**kwargs) for name, kwargs in cfg['val_augs'].items()])

    train_set = CommaDataset(cfg, 'train', train_img_augs, train_full_augs)
    val_set = CommaDataset(cfg, 'val', val_augs, None)
    
    print(f"trainset size: {len(train_set)}")
    print(f"valset size: {len(val_set)}")
    
    sample = train_set[0]
    print("First training set sample:")
    print(sample.keys())
    print(sample['frames'].shape)
    print(sample['frames'].dtype)
    print(sample['label_path'].shape)
    print(sample['label_path'].dtype)
    print(sample['prev_path'].shape)
    print(sample['prev_path'].dtype)

    np_img = logging.tensor_to_img(sample['frames'][0])
    disp_img = logging.display_path(np_img, sample['label_path'].numpy())

    plt.imshow(disp_img)
    plt.show()
