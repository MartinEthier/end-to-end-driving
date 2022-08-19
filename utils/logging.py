import numpy as np
import cv2
from data import transforms as dtf

from utils import paths
from lib.camera import FULL_FRAME_SIZE


def log_viz(frames, label, pred):
    """
    Display label and predictions on the input frames.

    frames: (B, T, 3, H, W)
    label: (B, N, 3)
    pred: (B, N, 3)
    """
    num_imgs = frames.shape[0]
    
    # Grab last frame in time sequence for viz
    np_frames = tensor_to_img(frames[:, -1]) # (B, H, W, 3)
    
    # Plot both paths on the images
    disp_imgs = []
    for i in range(num_imgs):
        disp_img = np_frames[i].copy()
        paths.draw_path(label[i], disp_img, width=0.15, fill_color=(0, 155, 0), line_color=(0, 255, 0))
        paths.draw_path(pred[i], disp_img, width=0.1, fill_color=(155, 0, 0), line_color=(255, 0, 0))
        disp_imgs.append(disp_img)

    return disp_imgs

def tensor_to_img(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert augmented tensor to numpy image for visualization.
    
    tensor: (..., 3, H, W)
    """
    # Denormalize tensor
    denormalize = dtf.Denormalize(mean, std)
    np_frames = np.moveaxis((denormalize(tensor).numpy() * 255), -3, -1).astype(np.uint8)
    
    return np_frames

def display_path(img, path):
    """
    Display a given path on the corresponding image.
    
    img: (H, W, 3) uint8 numpy array
    path: (N, 3) float numpy array
    """
    # Draw path on image
    paths.draw_path(path, display_img)
    
    return display_img
