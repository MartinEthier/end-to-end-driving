# Annoying ROS stuff
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from pathlib import Path
import argparse

import numpy as np
import cv2


def find_nearest(arr, target):
    """
    Given an input array and a target array, find the index in the input
    array of the nearest value for each of the target array values. Use 
    binary search to find the left value, then adjust the index if the
    right index is closer.
    """
    idx = np.searchsorted(arr, target)
    idx = np.clip(idx, 1, len(arr)-1)
    left = arr[idx-1]
    right = arr[idx]
    idx -= target - left < right - target
    return idx


class Route:
    """

    """
    def __init__(self, route_path):
        self.route_path = route_path

        # Stem of path is formatted: vehicle_id|route_timestamp. Split it to get timestamp
        self.route_time = route_path.stem.split('|')[-1]

        # Load in segments and sort in ascending order
        segments = [d for d in self.route_path.iterdir() if d.is_dir()]
        self.segment_paths = sorted(segments, key=lambda seg: int(seg.stem))

        # Init lists for accumulating segment arrays
        self.speeds = {'value': [], 't': []}
        self.angles = {'value': [], 't': []}
        self.frame_t = []

    def load_data(self):
        """
        Goes through each segment and loads in the np arrays for the speed 
        values, speed timestamps, angle values, angle timestamps, frame 
        images, and frame timestamps. Concatenates all arrays for the 
        segments together into 1 array for proper synchronization.
        """
        for segment_path in self.segment_paths:
            # Load in speed and angle arrays and add to accumulator list
            speed_dir = segment_path / "processed_log" / "CAN" / "speed"
            speed_t = np.load(speed_dir / "t")
            self.speeds['t'].append(speed_t)
            speed_value = np.load(speed_dir / "value")
            self.speeds['value'].append(np.squeeze(speed_value))
            
            angle_dir = segment_path / "processed_log" / "CAN" / "steering_angle"
            angle_t = np.load(angle_dir / "t")
            self.angles['t'].append(angle_t)
            angle_value = np.load(angle_dir / "value")
            self.angles['value'].append(angle_value)
            
            # Load in image time array and add to accumulator
            frame_t_path = segment_path / "global_pose" / "frame_times"
            frame_t_array = np.load(frame_t_path)
            self.frame_t.append(frame_t_array)

        # Concatenate all segments together
        for array_dict in [self.speeds, self.angles]:
            array_dict['t'] = np.concatenate(array_dict['t'])
            array_dict['value'] = np.concatenate(array_dict['value'])
        self.frame_t = np.concatenate(self.frame_t)
    
    def sync_arrays(self):
        """
        For each frame timestamp, find the nearest timestamp for speed 
        and angle. Only keep the speed/angle values that correspond with 
        those timestamps.
        """
        # Get idx that corresponds to each frame time and mask for those values
        speed_idx = find_nearest(self.speeds['t'], self.frame_t)
        self.synced_speed_value = self.speeds['value'][speed_idx]
        angle_idx = find_nearest(self.angles['t'], self.frame_t)
        self.synced_angle_value = self.angles['value'][angle_idx]

    def save_data(self, save_path):
        """
        Create new folder for route under the save_path directory and 
        save newly synced data + video frames to folder.
        """
        # Create save folder
        route_save_path = save_path / self.route_time
        route_save_path.mkdir()

        # Save synced arrays and timestamps
        np.save(route_save_path / 'speeds.npy', self.synced_speed_value)
        np.save(route_save_path / 'angles.npy', self.synced_angle_value)
        np.save(route_save_path / 'timestamps.npy', self.frame_t)

        # Load in each video and save the frames
        self.process_video(route_save_path)

    def process_video(self, route_save_path):
        """
        Go through video frames and save each as a png to the image
        directory. Name each according to index in timestamp array.
        """
        # Create images directory
        img_dir = route_save_path / "images"
        img_dir.mkdir()

        frame_count = 0
        # For each segment, load in the frames and save each one to the images dir
        for segment_path in self.segment_paths:
            video_path = segment_path / "video.hevc"

            cap = cv2.VideoCapture(str(video_path))
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Zero pad frame_count and save frame
                    img_path = img_dir / (str(frame_count).zfill(6) + '.png')
                    cv2.imwrite(str(img_path), frame)
                    frame_count += 1
                else:
                    break
            cap.release()


def main(root_path):
    # Create directory for processed dataset
    processed_dataset_path = root_path / "processed_dataset"
    processed_dataset_path.mkdir()

    # Loop through chunks for the Civic (3 to 10)
    for chunk_id in range(3, 11):
        print(f"Processing Chunk_{chunk_id}...")
        chunk_path = root_path / f"Chunk_{chunk_id}"

        # Loop through each route in the chunk
        route_paths = [f for f in chunk_path.iterdir() if f.is_dir()]
        for route_path in route_paths:
            # Create a Route object, load and sync the data, then save
            route = Route(route_path)
            print(f"Processing route {route.route_time}...")
            route.load_data()
            route.sync_arrays()
            print("Saving data...")
            route.save_data(processed_dataset_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='root directory of the comma2k19 dataset')
    args = parser.parse_args()

    root_path = Path(args.root_dir).expanduser()
    
    main(root_path)
