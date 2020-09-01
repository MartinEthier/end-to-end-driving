import torch
import numpy as np
import cv2


class Rescale(object):
    """
    Reshapes all frames to given size. Size is given as a tuple of (W, H).
    """
    def __init__(self, output_size):
        # Check output_size type
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        frames, angles, speeds = sample['frames'], sample['angles'], sample['speeds']
        
        # For each frame, resize to output_size and re-concat all into a 4d array
        resized_frames = []
        for i in range(frames.shape[0]):
            resized_frame = cv2.resize(frames[i], self.output_size, interpolation=cv2.INTER_AREA)
            resized_frames.append(resized_frame)
        resized_frames = np.stack(resized_frames)
            
        return {'frames': resized_frames,
                'angles': angles,
                'speeds': speeds}


class RandomHorizontalFlip(object):
    """
    Randomly flip frames horizontally with a certain probability. Adjust 
    steering angle accordingly if flipped. To be done before converting
    to torch tensor.
    """
    def __init__(self, prob):
        # Check prob type
        assert isinstance(prob, float)
        self.prob = prob

    def __call__(self, sample):
        frames, angles, speeds = sample['frames'], sample['angles'], sample['speeds']
        
        if np.random.uniform() < self.prob:
            # Flip frames horizontally, shape (T, H, W, C)
            frames = np.flip(frames, axis=2)
            
            # Adjust steering angles
            angles = angles * -1.0
            
        return {'frames': frames,
                'angles': angles,
                'speeds': speeds}


class Normalize(object):
    """
    
    """
    def __init__(self, norm_config):
        self.max_speed = norm_config["max_speed"]
        self.max_angle = norm_config["max_angle"]
        self.channel_mean = norm_config["channel_mean"]
        self.channel_std = norm_config["channel_std"]

    def __call__(self, sample):
        frames, angles, speeds = sample['frames'], sample['angles'], sample['speeds']
        
        # Normalize angles and speeds
        normalized_angles = angles / self.max_angle
        normalized_speeds = speeds / self.max_speed
        
        # Normalize images
        channel_mean = torch.DoubleTensor(self.channel_mean)
        channel_std = torch.DoubleTensor(self.channel_std)
        
        T, _, H, W = frames.shape
        mean_tensor = torch.transpose(channel_mean.repeat(W, H, 1), 0, 2).repeat(T, 1, 1).view(T, 3, H, W)
        std_tensor = torch.transpose(channel_std.repeat(W, H, 1), 0, 2).repeat(T, 1, 1).view(T, 3, H, W)
        
        normalized_frames = (frames - mean_tensor) / std_tensor
        
        return {'frames': normalized_frames,
                'angles': normalized_angles,
                'speeds': normalized_speeds}


class ToTensor(object):
    """
    Convert ndarrays in sample to tensors.
    """
    def __call__(self, sample):
        frames, angles, speeds = sample['frames'], sample['angles'], sample['speeds']
        
        # Normalize to be between 0 and 1
        frames = (frames - np.amin(frames))/np.amax(frames - np.amin(frames)).astype(np.float64)

        # Swap frame axes from (T, H, W, C) to (T, C, H, W) for torch tensor
        frames = frames.transpose((0, 3, 1, 2))
        
        return {'frames': torch.from_numpy(frames),
                'angles': torch.from_numpy(angles),
                'speeds': torch.from_numpy(speeds)}

