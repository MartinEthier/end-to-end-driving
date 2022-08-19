from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from utils import paths


def main(args, train_files):
    """
    Calculate min, max, mean, std for all speeds in dataset and generate histogram.
    Generate all the label paths for the dataset and calculate the mean for each timestep
    for bias initialization.
    """
    # Group file by route
    route_dict = {}
    curvatures = []
    for img_file, curvature in train_files:
        curvatures.append(curvature)
        img_path = Path(img_file)
        route = img_path.parent.parent.stem
        img_idx = int(img_path.stem)
        if route not in route_dict:
            route_dict[route] = []
        route_dict[route].append(img_idx)
        
    print(route_dict.keys())
    print(len(route_dict[list(route_dict)[0]]))


    future_path_accum = np.zeros((args['future_steps'], 3), dtype=np.float32)
    print(future_path_accum.shape)
    num_paths = 0
    future_paths = []
    for route, img_idxs in route_dict.items():
        # Load route data arrays
        route_path = Path(args['root_dir']) / route
        orientations = np.load(route_path / "frame_orientations.npy")
        positions = np.load(route_path / "frame_positions.npy")

        for img_idx in img_idxs:
            # Convert positions to reference frame
            local_path = paths.get_local_path(positions, orientations, img_idx)
            future_path = local_path[img_idx + 1 : img_idx + 1 + args['future_steps']]
            #print(future_path.shape)
            future_path_accum += future_path
            num_paths += 1
            print(num_paths)

            

    mean_path = future_path_accum / num_paths
    s = mean_path.shape
    print(mean_path)
    print(mean_path.reshape(s[0]*s[1]))
    mean_path = mean_path.reshape(s[0]*s[1])
    np.savetxt(Path(__file__).parent / "dataset_lists" / "mean_path.txt", mean_path)






    # plt.hist(curvatures, bins='auto')
    # plt.show()

    """
    route_paths = [f for f in root_path.iterdir() if f.is_dir()]
    
    # Loop through routes and accumulate speeds + angles
    all_speeds = []
    for route_path in route_paths:
        print(route_path.stem)
        speeds = np.load(route_path / "CAN_speeds.npy")
        
        all_speeds.append(speeds)

    all_speeds = np.concatenate(all_speeds)
    
    # Print size of dataset
    print(f"Dataset size: {all_speeds.shape[0]}")
    
    # Print out min, max, mean, and median values for both variables
    print(f"Speeds:\nMin: {np.min(all_speeds)}\nMax: {np.max(all_speeds)}\nMean: {np.mean(all_speeds)}\nMedian: {np.median(all_speeds)}")
    
    # Generate histograms
    fig, axes = plt.subplots(1, 1)
    axes[0].hist(all_speeds, bins=100)
    plt.show()
    """
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', type=Path, help='Name of the dataset list file.')
    args = parser.parse_args()

    # Load dataset file
    json_path = Path(__file__).parent / 'dataset_lists' / args.dataset_file
    with json_path.open('r') as fr:
        trainval_dict = json.load(fr)
    print(trainval_dict.keys())
    print(trainval_dict['args'])
    print(trainval_dict['train_files'][0])

    main(trainval_dict['args'], trainval_dict['train_files'])
