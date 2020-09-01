from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt



def main():
    """
    Loop through all routes in processed dataset and accumulate speeds/angles. 
    Then generate histograms for the values and print the max, min, and mean 
    values for each variable.
    """
    dataset_path = Path("/media/watouser/Seagate_Backup/comma2k19/processed_dataset")
    route_paths = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    # Loop through routes and accumulate speeds + angles
    all_angles = []
    all_speeds = []
    for route_path in route_paths:
        print(route_path.stem)
        angles = np.load(route_path / "angles.npy")
        speeds = np.load(route_path / "speeds.npy")
        
        all_angles.append(angles)
        all_speeds.append(speeds)
        
    all_angles = np.concatenate(all_angles)
    all_speeds = np.concatenate(all_speeds)
    
    # Print size of dataset
    print(f"Dataset size: {all_angles.shape[0]}")
    
    # Print out min, max, mean, and median values for both variables
    print(f"Speeds:\nMin: {np.min(all_speeds)}\nMax: {np.max(all_speeds)}\nMean: {np.mean(all_speeds)}\nMedian: {np.median(all_speeds)}")
    print(f"Angles:\nMin: {np.min(all_angles)}\nMax: {np.max(all_angles)}\nMean: {np.mean(all_angles)}\nMedian: {np.median(all_angles)}")
    
    # Generate histograms
    histogram_path = dataset_path / "statistic_histograms"
    histogram_path.mkdir()
    
    fig, ax = plt.subplots()
    ax.hist(all_speeds, bins=100)
    fig.savefig(histogram_path / "speeds_histogram.png")
    
    fig, ax = plt.subplots()
    ax.hist(all_angles, bins=100)
    fig.savefig(histogram_path / "angles_histogram.png")
    
    
if __name__=="__main__":
    main()
    



