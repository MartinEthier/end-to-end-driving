from pathlib import Path
import random

import torch
import cv2
import numpy as np

import time

class CommaDataset(torch.utils.data.Dataset):
    """

    """

    def __init__(self, dataset_cfg, transform=None):
        """
        dataset_cfg:
        """
        root_path = Path(dataset_cfg['root_path'])
        self.seq_len = dataset_cfg['seq_len']
        self.transform = transform

        # Accumulate list of paths to each image frame in dataset
        self.frame_paths = []
        route_paths = [f for f in root_path.iterdir() if f.is_dir()]
        for route_path in route_paths:
            # Get image paths
            image_paths = list((route_path / 'images').glob('*.jpg'))

            # Skip route if too few images to make a sequence
            if len(image_paths) < self.seq_len:
                continue
            
            # Remove the frames which have less than seq_len-1 previous frames
            del image_paths[:self.seq_len-1]

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
        frame_path = self.frame_paths[idx]
        route_path = frame_path.parent.parent
        frame_id = int(frame_path.stem)
        
        # Load angles and speeds
        angles_path = route_path / "angles.npy"
        speeds_path = route_path / "speeds.npy"
        angles = np.load(angles_path)
        speeds = np.load(speeds_path)

        # Index to current + previous in sequence
        angles = angles[frame_id - self.seq_len + 1:frame_id + 1]
        speeds = speeds[frame_id - self.seq_len + 1:frame_id + 1]

        # Loop through frame and previous frames and add them together
        frames = []
        for f_id in range(frame_id - self.seq_len + 1, frame_id + 1):
            filename = str(f_id).zfill(6) + '.jpg'
            frame = cv2.imread(str(route_path / "images" / filename), cv2.IMREAD_COLOR)
            frames.append(frame)
        
        # Stack frames into single array, (T, H, W, C)
        frames = np.stack(frames)

        sample = {}
        sample['angles'] = angles
        sample['speeds'] = speeds
        sample['frames'] = frames
        
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__=="__main__":
    cfg = {'root_path': "/media/watouser/Seagate_Backup/comma2k19/processed_dataset",
            'seq_len': 10,
            'size': 20000,
            'incl_speed': False,
            }
    comma_ds = CommaDataset(cfg)
    print(len(comma_ds))

