import numpy as np
import cv2

from utils import paths


def gen_logs(log_tensors):
    # Move channels to last dim
    img = np.transpose(log_tensors['img'], (1, 2, 0))
    # Normalize to a 0-255 uint8 and resize to original dataset size
    img = ((img - np.amin(img)) * (1/(np.amax(img) - np.amin(img)) * 255)).astype('uint8')
    img = cv2.resize(img, (1164, 874))

    # Resize paths to (N, 3)
    label = np.reshape(log_tensors['label'], (-1, 3))
    output = np.reshape(log_tensors['output'], (-1, 3))

    # Draw paths on image
    label_img = img.copy()
    pred_img = img.copy()
    paths.draw_path(label, label_img)
    paths.draw_path(output, pred_img)

    # Add both paths to a 3d plot
    fig = paths.plot_3d_paths(label, output)

    return label_img, pred_img, fig
