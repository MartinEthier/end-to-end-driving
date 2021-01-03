import numpy as np
import cv2
import plotly.express as px

from utils import paths


def gen_logs(log_tensors, norm_params):
    img = np.transpose(log_tensors['img'], (1, 2, 0))
    img = ((img - np.amin(img)) * (1/(np.amax(img) - np.amin(img)) * 255)).astype('uint8')
    img = cv2.resize(img, (1164, 874))
    
    label = np.reshape(log_tensors['label'], (-1, 3))
    paths.descale(label, norm_params)
    
    output = np.reshape(log_tensors['output'], (-1, 3))
    paths.descale(output, norm_params)
    
    label_img = img.copy()
    pred_img = img.copy()

    paths.draw_path(label, label_img)
    paths.draw_path(output, pred_img)

    fig = px.line_3d(x=np.concatenate([label[:, 0], output[:, 0]]),
                     y=np.concatenate([label[:, 1], output[:, 1]]),
                     z=np.concatenate([label[:, 2], output[:, 2]]),
                     color=['label'] * label.shape[0] + ['pred'] * output.shape[0]
                    )
    
    return label_img, pred_img, fig
