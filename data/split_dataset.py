"""

"""
from pathlib import Path
import argparse
import random
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from utils import paths

random.seed(0)

MIN_SPEED = 10 # m/s
GRAD_SPIKE_THRESH = 0.05


def get_route_frames(route_path, future_steps, past_steps):
    print(f"Processing route {route_path}")
    image_files = []

    # Load route data arrays
    velocities = np.load(route_path / "frame_velocities.npy")
    steering_angles = np.load(route_path / "CAN_angles.npy")
    orientations = np.load(route_path / "frame_orientations.npy")
    positions = np.load(route_path / "frame_positions.npy")

    # Get image paths and sort by filename
    image_paths = list((route_path / 'images').glob('*.jpg'))
    image_paths.sort(key=lambda path: path.stem)

    # Skip route if too few images to make a sequence
    if len(image_paths) < past_steps + future_steps + 1:
        return ([], [])

    # Remove the frames that don't have enough previous or future frames
    del image_paths[-future_steps:]
    del image_paths[:past_steps]

    for image_path in image_paths:
        # Get frame id
        frame_id = int(image_path.stem)
        
        # Remove examples where the car is going backwards
        future_x_vel = velocities[frame_id + 1 : frame_id + 1 + future_steps, 0]
        if np.any(future_x_vel < 0):
            continue
            
        # Remove examples with an avg speed below MIN_SPEED
        avg_speed = np.mean(np.linalg.norm(velocities[frame_id + 1 : frame_id + 1 + future_steps], axis=1))
        if avg_speed < MIN_SPEED:
            continue
            
        # Convert positions to reference frame
        local_path = paths.get_local_path(positions, orientations, frame_id)
        
        # Get full path
        full_path = local_path[frame_id - past_steps : frame_id + 1 + future_steps]
        
        # Remove examples where the path has a glitch in it by getting path gradients and finding spikes
        grads = np.mean(np.gradient(full_path, axis=0), axis=1)
        grad_diffs = np.abs(np.diff(grads))
        if np.any(grad_diffs > GRAD_SPIKE_THRESH):
            continue
        
        # Get future path
        future_x_path = local_path[frame_id + 1 : frame_id + 1 + future_steps, 0]
        
        # Remove examples with any negative X positions
        if np.any(future_x_path < 0):
            continue
        
        # Get curvature score by taking RMSE between steering angles through future path and initial steering angle
        # Actually should just be the magnitude of the steering angle, 
        # Can either take sum of abs(angles) or sum of angles**2
        # future_angles = steering_angles[frame_id + 1 : frame_id + 1 + future_steps]
        # ref_angle = steering_angles[frame_id]
        # curvature_score = np.sqrt(np.sum((future_angles-ref_angle)**2) / future_angles.shape[0])

        curvature_score = np.mean(np.abs(steering_angles[frame_id : frame_id + future_steps]))

        image_files.append((image_path, curvature_score))

    return image_files

def main(args):
    # Accumulate list of paths to each image frame in dataset
    train_files = []
    val_files = []
    route_paths = [f for f in args.root_dir.iterdir() if f.is_dir()]
    random.shuffle(route_paths)
    
    split_idx = int(args.trainval_split * len(route_paths))
    train_paths = route_paths[:split_idx]
    val_paths = route_paths[split_idx:]
    print(f"train routes: {len(train_paths)}, val routes: {len(val_paths)}")

    # Loop through specified chunks
    with ProcessPoolExecutor() as executor:
        train_results = executor.map(get_route_frames, train_paths, itertools.repeat(args.future_steps), itertools.repeat(args.past_steps))
    for res in train_results:
        train_files.extend(res)
    train_files = [(str(file[0]), file[1]) for file in train_files]
    with ProcessPoolExecutor() as executor:
        val_results = executor.map(get_route_frames, val_paths, itertools.repeat(args.future_steps), itertools.repeat(args.past_steps))
    for res in val_results:
        val_files.extend(res)
    val_files = [(str(file[0]), file[1]) for file in val_files]
        
    print(f"train images: {len(train_files)}, val images: {len(val_files)}")
    
    # # Get list of all curvatures in training set and bin them
    # curvature_list = [ex[1] for ex in train_files]
    # print(len(curvature_list))
    # print(curvature_list[0])
    # curvature_array = np.array(curvature_list)
    # print(curvature_array.shape)
      
    # Convert Path to str and save to json
    arg_dict = vars(args)
    arg_dict['root_dir'] = str(arg_dict['root_dir'])
    arg_dict['train_size'] = len(train_files)
    arg_dict['val_size'] = len(val_files)
    arg_dict['train_routes'] = len(train_paths)
    arg_dict['val_routes'] = len(val_paths)
    save_data = {'args': arg_dict, 'train_files': train_files, 'val_files': val_files}
    save_json_path = Path(__file__).parent / 'dataset_lists/full_trainval_set_curv.json'
    with save_json_path.open('w') as fs:
        json.dump(save_data, fs)
        
    print(f"trainval set saved at: {str(save_json_path)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=Path, help='Root directory of the preprocessed dataset')
    parser.add_argument("-f", "--future_steps", type=int, help="The number of steps to predict into the future")
    parser.add_argument("-p", "--past_steps", type=int, help="The number of past steps to use when predicting")
    parser.add_argument("-s", "--trainval_split", type=float, help="Ratio of training data in trainval split")
    args = parser.parse_args()

    # Check args
    if not args.root_dir.exists():
        raise AssertionError("Root directory does not exist", args.root_dir)
    if args.future_steps < 1:
        raise AssertionError("Future steps less than 1", args.future_steps)
    if args.past_steps < 0:
        raise AssertionError("Past steps less than 0", args.past_steps)
    if args.trainval_split < 0 or args.trainval_split > 1:
        raise AssertionError("Trainval split not between 0 and 1", args.trainval_split)

    main(args)
