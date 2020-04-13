import torch
import numpy as np






class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, angle, speed = sample['image'], sample['angle'], sample['speed']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'angle': torch.from_numpy(angle),
                'speed': torch.from_numpy(speed)}

