from pathlib import Path
import json

import torch
import cv2
import numpy as np
from PIL import Image

from utils.paths import get_local_path


class CommaDataset(torch.utils.data.Dataset):
    """

    """

    def __init__(self, cfg, split, img_transform, data_transform=None):
        """
        dataset_cfg:
        """
        for key, value in cfg.items():
            setattr(self, key, value)
        self.img_transform = img_transform
        self.data_transform = data_transform
        
        # Load in trainval json
        json_path = Path(__file__).parent / 'dataset_lists' / self.dataset_file
        with json_path.open('r') as fr:
            trainval_dict = json.load(fr)
        self.args = trainval_dict['args']
        
        # Set files and dataset size based on split
        train_size = int(self.args['dataset_size'] * self.args['trainval_split'])
        if split == 'train':
            self.dataset_size = train_size
            self.frame_paths = [Path(f_path) for f_path in trainval_dict['train_set']]
        elif split == 'val':
            self.dataset_size = self.args['dataset_size'] - train_size
            self.frame_paths = [Path(f_path) for f_path in trainval_dict['val_set']]
        else:
            raise ValueError("Invalid split string", self.split)

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
        local_path = get_local_path(positions, orientations, frame_id)
        
        # Divide data into previous and future arrays
        future_path = local_path[frame_id + 1 : frame_id + 1 + self.args['future_steps']]
        previous_path = local_path[frame_id - self.args['past_steps'] : frame_id + 1]
        
        # Grab previous and current frames
        frames = []
        for f_id in range(frame_id - self.args['past_steps'], frame_id + 1):
            filename = str(f_id).zfill(6) + '.jpg'
            frame = Image.open(str(route_path / "images" / filename))
            
            # Apply transforms to frame
            frame = self.img_transform(frame)
            frames.append(frame)
        
        # Stack frames into single array, (T, C, H, W)
        frames = torch.stack(frames)

        sample = {}
        sample['frames'] = frames
        sample['label_path'] = torch.from_numpy(future_path)
        sample['prev_path'] = torch.from_numpy(previous_path)
        
        if self.predict_speed:
            # Load in velocities and convert to speed
            velocities = np.load(route_path / "frame_velocities.npy")
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Divide data into previous and future arrays
            future_speeds = speeds[frame_id + 1 : frame_id + 1 + self.args['future_steps']]
            previous_speeds = speeds[frame_id - self.args['past_steps'] : frame_id + 1]
            
            sample['label_speed'] = torch.from_numpy(future_speeds)
            sample['prev_speed'] = torch.from_numpy(previous_speeds)
            
        # Apply transforms to path and speed arrays
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample


if __name__=="__main__":
    cfg = {"dataset_file": "trainval_set.json",
           "split": "val",
           "predict_speed": True
            }
    
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    
    img_transforms = Compose([
        Resize([384, 288]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    comma_ds = CommaDataset(cfg, img_transforms)
    sample = comma_ds[0]
    for key, val in sample.items():
        print(key)
        print(val.shape)
    print(f"dataset size: {len(comma_ds)}")
