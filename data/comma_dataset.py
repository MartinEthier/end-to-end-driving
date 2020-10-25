from pathlib import Path
import random
import time

import torch
import cv2
import numpy as np

from utils.paths import get_local_path


class CommaDataset(torch.utils.data.Dataset):
    """

    """

    def __init__(self, dataset_cfg, transform=None):
        """
        dataset_cfg:
        """
        for key, value in dataset_cfg.items():
            setattr(self, key, value)
        self.root_path = Path(self.root_path)
        self.transform = transform

        # Accumulate list of paths to each image frame in dataset
        self.frame_paths = []
        route_paths = [f for f in self.root_path.iterdir() if f.is_dir()]
        for route_path in route_paths:
            # Get image paths
            image_paths = list((route_path / 'images').glob('*.jpg'))

            # Skip route if too few images to make a sequence
            if len(image_paths) < self.past_steps + self.predict_steps + 1:
                continue
            
            # Remove the frames that don't have enough previous or future frames
            del image_paths[-self.predict_steps:]
            del image_paths[:self.past_steps]

            # Add paths to accumulator list
            self.frame_paths.extend(image_paths)

        # Shuffle the paths
        random.seed(0)
        random.shuffle(self.frame_paths)
        
        # Sample the dataset to match config size
        if 'size' in dataset_cfg:
            self.frame_paths = self.frame_paths[0:dataset_cfg['size']]

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Get paths and frame id
        img_path = self.frame_paths[idx]
        route_path = img_path.parent.parent
        frame_id = int(img_path.stem)
        
        # Load route data arrays
        orientations = np.load(route_path / "frame_orientations.npy")
        positions = np.load(route_path / "frame_positions.npy")
        velocities = np.load(route_path / "frame_velocities.npy")
        
        # Convert positions to reference frame
        local_path = get_local_path(positions, orientations, frame_id)
        
        # Divide data into previous and future arrays
        future_path = local_path[frame_id + 1 : frame_id + 1 + self.predict_steps]
        previous_path = local_path[frame_id - self.past_steps : frame_id]
        future_velocities = velocities[frame_id + 1 : frame_id + 1 + self.predict_steps]
        previous_velocities = velocities[frame_id - self.past_steps : frame_id]
        
        # Grab previous and current frames
        frames = []
        for f_id in range(frame_id - self.past_steps, frame_id + 1):
            filename = str(f_id).zfill(6) + '.jpg'
            frame = cv2.imread(str(route_path / "images" / filename), cv2.IMREAD_COLOR)
            
            # ?
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # Stack frames into single array, (T, H, W, C)
        frames = np.stack(frames)

        sample = {}
        sample['frames'] = frames
        sample['label_path'] = future_path
        sample['prev_path'] = previous_path
        sample['label_vel'] = future_velocities
        sample['prev_vel'] = previous_velocities
        
        # ?
        if self.transform:
            frame = self.transform(frame)

        return sample


if __name__=="__main__":
    cfg = {"root_path": "/media/watouser/Seagate_Backup/datasets/comma2k19/processed_dataset",
           "past_steps": 5,
           "predict_steps": 60,
           "size": 10000,
           "predict_speed": False
            }
    comma_ds = CommaDataset(cfg)
    sample = comma_ds[0]
