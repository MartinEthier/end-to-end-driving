import numpy as np
import cv2

import lib.orientation as orient


def get_local_path(positions, orientations, reference_idx):
    # Define the segment frame positions referenced at a desired frame
    ecef_from_local = orient.rot_from_quat(orientations[reference_idx])
    local_from_ecef = ecef_from_local.T
    positions_local = np.einsum('ij,kj->ki', local_from_ecef, positions - positions[reference_idx])
    return positions_local




