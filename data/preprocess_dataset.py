"""
Run this script to generate jpg images for each frame in the dataset and create 
numpy arrays for timestamps, steering angles, and vehicles speeds. The dataset
is organized by route. This is intended to be used for training end-to-end 
control models. The script also organizes the global frame times, positions,
velocities, and orientations to be used with the comma library to generate paths
for end-to-end planner training.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

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
    def __init__(self, route_path, frame_size):
        self.route_path = route_path
        self.frame_size = frame_size

        # Stem of path is formatted: vehicle_id|route_timestamp. Split it to get timestamp
        self.route_time = route_path.stem.split('|')[-1]

        # Load in segments and sort in ascending order
        segments = [d for d in self.route_path.iterdir() if d.is_dir()]
        self.segment_paths = sorted(segments, key=lambda seg: int(seg.stem))

        # Init lists for accumulating segment arrays
        self.CAN_speeds = {'value': [], 't': []}
        self.CAN_angles = {'value': [], 't': []}
        self.frame_t = []
        self.frame_pos = []
        self.frame_vel = []
        self.frame_ori = []

    def load_data(self):
        """
        Goes through each segment and loads in the np arrays for the speed 
        values, speed timestamps, angle values, angle timestamps, frame 
        images, frame timestamps, frame positions, frame velocities, and frame 
        orientations. Concatenates all arrays for the segments together into 1 
        array for proper synchronization.
        """
        for segment_path in self.segment_paths:
            # Load in speed and angle arrays and add to accumulator list
            speed_dir = segment_path / "processed_log" / "CAN" / "speed"
            speed_t = np.load(speed_dir / "t")
            self.CAN_speeds['t'].append(speed_t)
            speed_value = np.load(speed_dir / "value")
            self.CAN_speeds['value'].append(np.squeeze(speed_value))
            
            angle_dir = segment_path / "processed_log" / "CAN" / "steering_angle"
            angle_t = np.load(angle_dir / "t")
            self.CAN_angles['t'].append(angle_t)
            angle_value = np.load(angle_dir / "value")
            self.CAN_angles['value'].append(angle_value)
            
            # Load in global_pose arrays and add to accumulator
            frame_t_path = segment_path / "global_pose" / "frame_times"
            frame_t_array = np.load(frame_t_path)
            self.frame_t.append(frame_t_array)
            
            frame_pos_path = segment_path / "global_pose" / "frame_positions"
            frame_pos_array = np.load(frame_pos_path)
            self.frame_pos.append(frame_pos_array)
            
            frame_vel_path = segment_path / "global_pose" / "frame_velocities"
            frame_vel_array = np.load(frame_vel_path)
            self.frame_vel.append(frame_vel_array)
            
            frame_ori_path = segment_path / "global_pose" / "frame_orientations"
            frame_ori_array = np.load(frame_ori_path)
            self.frame_ori.append(frame_ori_array)

        # Concatenate all segments together
        for array_dict in [self.CAN_speeds, self.CAN_angles]:
            array_dict['t'] = np.concatenate(array_dict['t'])
            array_dict['value'] = np.concatenate(array_dict['value'])
        self.frame_t = np.concatenate(self.frame_t)
        self.frame_pos = np.concatenate(self.frame_pos)
        self.frame_vel = np.concatenate(self.frame_vel)
        self.frame_ori = np.concatenate(self.frame_ori)
    
    def sync_arrays(self):
        """
        For each frame timestamp, find the nearest timestamp for speed 
        and angle. Only keep the speed/angle values that correspond with 
        those timestamps.
        """
        # Get idx that corresponds to each frame time and mask for those values
        speed_idx = find_nearest(self.CAN_speeds['t'], self.frame_t)
        self.synced_speed_value = self.CAN_speeds['value'][speed_idx]
        angle_idx = find_nearest(self.CAN_angles['t'], self.frame_t)
        self.synced_angle_value = self.CAN_angles['value'][angle_idx]

    def save_data(self, save_path):
        """
        Create new folder for route under the save_path directory and 
        save newly synced data + video frames to folder.
        """
        # Create save folders
        route_save_path = save_path / self.route_time
        route_save_path.mkdir()

        # Save synced arrays and global_pose arrays
        np.save(route_save_path / 'CAN_speeds.npy', self.synced_speed_value)
        np.save(route_save_path / 'CAN_angles.npy', self.synced_angle_value)
        np.save(route_save_path / 'frame_times.npy', self.frame_t)
        np.save(route_save_path / 'frame_positions.npy', self.frame_pos)
        np.save(route_save_path / 'frame_velocities.npy', self.frame_vel)
        np.save(route_save_path / 'frame_orientations.npy', self.frame_ori)

        # Load in each video and save the frames
        self.process_video(route_save_path)

    def process_video(self, route_save_path):
        """
        Go through video frames and save each as a jpg to the image
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
                    # Resize frame to target size
                    resized = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

                    # Zero pad frame_count and save frame
                    img_path = img_dir / (str(frame_count).zfill(6) + '.jpg')
                    cv2.imwrite(str(img_path), resized)
                    frame_count += 1
                else:
                    break
            cap.release()


def process_chunk(root_path, processed_dataset_path, frame_size, chunk_id):
    # Loop through each route in the chunk
    chunk_path = root_path / f"Chunk_{chunk_id}"
    route_paths = [f for f in chunk_path.iterdir() if f.is_dir()]
    for route_path in route_paths:
        # Create a Route object, load and sync the data, then save
        route = Route(route_path, frame_size)
        print(f"Processing route {route.route_time}...")
        route.load_data()
        route.sync_arrays()
        route.save_data(processed_dataset_path)

def main(root_path, chunk_range, frame_size):
    # Create directory for processed dataset
    processed_dataset_path = root_path / "processed_dataset"
    processed_dataset_path.mkdir()

    # Loop through specified chunks
    with ProcessPoolExecutor() as executor:
        for chunk_id in range(chunk_range[0], chunk_range[1] + 1):
            executor.submit(process_chunk, root_path, processed_dataset_path, frame_size, chunk_id)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='Root directory of the comma2k19 dataset.')
    parser.add_argument('-s', '--frame_size', type=int, nargs=2, default=[384, 288], help='Size to reshape all the frames to.')
    parser.add_argument('-c', '--chunk_range', type=int, nargs=2, default=[3, 10], help='Which range of chunks to process (default is 3-10 for Honda Civic data).')
    args = parser.parse_args()

    root_path = Path(args.root_dir).expanduser()

    main(root_path, args.chunk_range, args.frame_size)
