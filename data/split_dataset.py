"""

"""

# Annoying ROS stuff
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from pathlib import Path
import argparse
import random
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

random.seed(0)


def get_route_frames(route_path, future_steps, past_steps):
    image_files = []
    curvature_scores = []

    # Load route data arrays
    velocities = np.load(route_path / "frame_velocities.npy")
    steering_angles = np.load(route_path / "CAN_angles.npy")

    # Get image paths
    image_paths = list((route_path / 'images').glob('*.jpg'))

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
        if np.any(future_x_vel <= 0):
            continue
            
        # Remove examples with an avg speed below 10 m/s
        avg_speed = np.mean(np.linalg.norm(velocities[frame_id + 1 : frame_id + 1 + future_steps], axis=1))
        if avg_speed < 10:
            continue
            
        # Get future steering angles
        future_angles = steering_angles[frame_id + 1 : frame_id + 1 + future_steps]
        
        # Get curvature score by taking RMSE between steering angles through path and initial steering angle
        ref_angle = steering_angles[frame_id]
        curvature_score = np.sqrt(np.sum((future_angles-ref_angle)**2) / future_angles.shape[0])
        
        image_files.append(image_path)
        curvature_scores.append(curvature_score)

    return (image_files, curvature_scores)


def main(args):
    # Define route for test set
    test_json_path = Path(__file__).parent / 'dataset_lists/test_set_routes.json'
    with test_json_path.open('r') as fr:
        test_routes = json.load(fr)['test_routes']

    # Accumulate list of paths to each image frame in dataset
    image_files = []
    curvature_scores = []
    route_paths = [f for f in args.root_dir.iterdir() if f.is_dir() and f.stem not in test_routes]

    # Loop through specified chunks
    with ProcessPoolExecutor() as executor:
        results = executor.map(get_route_frames, route_paths, itertools.repeat(args.future_steps), itertools.repeat(args.past_steps))
    for result in results:
        image_files.extend(result[0])
        curvature_scores.extend(result[1])

    curvature_array = np.array(curvature_scores)

    print(f"Dataset size before binning: {curvature_array.shape[0]}")

    dataset_files = []
    ds_curvatures = []

    _, bin_edges = np.histogram(curvature_array, bins=args.num_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_counts = np.zeros(args.num_bins)

    while len(dataset_files) < args.dataset_size and len(image_files) != 0:
        # Sample a random frame
        index = random.randrange(len(image_files))
        img_file = image_files.pop(index)
        curvature = curvature_scores.pop(index)

        # Find the bin it falls in
        frame_bin = np.argmin(np.abs(bin_centers - curvature))

        # Check if bin is full, if full skip to next example
        if bin_counts[frame_bin] == args.max_bin_size:
            continue

        # Collect example and increment bin count
        dataset_files.append(img_file)
        ds_curvatures.append(curvature)
        bin_counts[frame_bin] += 1

    # Split dataset into train/val
    dataset_files = [str(file) for file in dataset_files]
    random.shuffle(dataset_files)
    train_set = dataset_files[:int(args.dataset_size*args.trainval_split)]
    val_set = dataset_files[int(args.dataset_size*args.trainval_split):]
    
    print(f"train size: {len(train_set)}, val size: {len(val_set)}")
    
    # Convert Path to str and save to json
    arg_dict = vars(args)
    arg_dict['root_dir'] = str(arg_dict['root_dir'])
    save_data = {'args': arg_dict, 'train_set': train_set, 'val_set': val_set}
    save_json_path = Path(__file__).parent / 'dataset_lists/trainval_set.json'
    with save_json_path.open('w') as fs:
        json.dump(save_data, fs)
        
    print(f"trainval set saved at: {str(save_json_path)}")
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=Path, help='Root directory of the preprocessed dataset')
    parser.add_argument("-f", "--future_steps", type=int, help="The number of steps to predict into the future")
    parser.add_argument("-p", "--past_steps", type=int, help="The number of past steps to use when predicting")
    parser.add_argument("-d", "--dataset_size", type=int, help="Number of examples in the trainval set")
    parser.add_argument("-m", "--max_bin_size", type=int, help="Max number of examples in each curvature bin")
    parser.add_argument("-b", "--num_bins", type=int, help="Number of curvature bins to use")
    parser.add_argument("-s", "--trainval_split", type=float, help="Ratio of training data in trainval split")
    args = parser.parse_args()

    # Check args
    if not args.root_dir.exists():
        raise AssertionError("Root directory does not exist", args.root_dir)
    if args.future_steps < 1:
        raise AssertionError("Future steps less than 1", args.future_steps)
    if args.past_steps < 0:
        raise AssertionError("Past steps less than 0", args.past_steps)
    if args.dataset_size < 1:
        raise AssertionError("Dataset size than 1", args.dataset_size)
    if args.max_bin_size < 1:
        raise AssertionError("Max bin size less than 1", args.max_bin_size)
    if args.num_bins < 1:
        raise AssertionError("Num bins less than 1", args.num_bins)
    if args.trainval_split < 0 or args.trainval_split > 1:
        raise AssertionError("Trainval split not between 0 and 1", args.trainval_split)

    main(args)
