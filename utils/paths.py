import numpy as np
import cv2
import plotly.express as px
from scipy import interpolate

import lib.orientation as orient
import lib.camera as cam
from lib.camera import img_from_device, denormalize


def get_local_path(positions, orientations, reference_idx):
    # Define the segment frame positions referenced at a desired frame
    ecef_from_local = orient.rot_from_quat(orientations[reference_idx])
    local_from_ecef = ecef_from_local.T
    positions_local = np.einsum('ij,kj->ki', local_from_ecef, positions - positions[reference_idx])

    return positions_local

def draw_path(device_path, img, width=0.5, height=1, fill_color=(128,0,255), line_color=(0,255,0)):
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:,1] -= width
    device_path_r[:,1] += width

    img_points_norm_l = cam.img_from_device(device_path_l)
    img_points_norm_r = cam.img_from_device(device_path_r)
    img_pts_l = cam.denormalize(img_points_norm_l)
    img_pts_r = cam.denormalize(img_points_norm_r)

    # filter out things rejected along the way
    valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
    img_pts_l = img_pts_l[valid].astype(int)
    img_pts_r = img_pts_r[valid].astype(int)

    for i in range(1, len(img_pts_l)):
        u1,v1,u2,v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
        u3,v3,u4,v4 = np.append(img_pts_l[i], img_pts_r[i])
        pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], fill_color)
        cv2.polylines(img, [pts], True, line_color)

def plot_3d_paths(label_path, pred_path):
    # Plot both the label path and predicted path in a 3D plot
    fig = px.line_3d(
        x=np.concatenate([label_path[:, 0], pred_path[:, 0]]),
        y=np.concatenate([label_path[:, 1], pred_path[:, 1]]),
        z=np.concatenate([label_path[:, 2], pred_path[:, 2]]),
        color=['label'] * label_path.shape[0] + ['pred'] * pred_path.shape[0]
    )

    return fig

def smooth_path(path):
    # Interpolate path
    tck, u = interpolate.splprep([path[:, 0], path[:, 1], path[:, 2]], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, path.shape[0])
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    # Reconcat
    smooth_path = np.concatenate((x_fine[:, np.newaxis], y_fine[:, np.newaxis], z_fine[:, np.newaxis]), axis=1)
    
    return smooth_path
