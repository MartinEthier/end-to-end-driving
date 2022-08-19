import random

import torch
import torchvision.transforms as tf
import numpy as np
import cv2


class Denormalize():
    """
    Inverse of the normalization with the provided mean and std params.
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.denormalize = tf.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std]
        )
    
    def __call__(self, tensor):
        return self.denormalize(tensor)


class RandomHorizontalFlip():
    """
    Randomly flip frames horizontally with a certain probability. Adjust paths
    accordingly when flipped.
    """
    def __init__(self, prob):
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        
        self.flip = tf.RandomHorizontalFlip(p=1.0)

    def __call__(self, sample):
        """
        sample keys:
        frames: (T, C, H, W)
        label_path: (N, 3)
        prev_path: (N, 3)
        """
        frames, label_path, prev_path = sample['frames'], sample['label_path'], sample['prev_path']
        
        if np.random.uniform() < self.prob:
            frames = self.flip(frames)
            label_path[:, 1] *= -1
            prev_path[:, 1] *= -1
            
        return {'frames': frames,
                'label_path': label_path,
                'prev_path': prev_path}
