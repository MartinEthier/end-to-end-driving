# end-to-end-driving

Repository to train end-to-end neural nets on the [comma2k19 dataset](https://github.com/commaai/comma2k19).

## Environment setup
Setup a conda environment with the libraries specified in requirements.txt

## Dataset setup
1. Download the dataset from [here](https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb) using the wget command and then run a command-line torrent client on the downloaded file to download the full dataset (I used rTorrent).
2. Run the pre-processing script as follows. Chunk range is set to 3-10 by default (Civic). Set to 1-2 for Rav4.
```bash
python preprocess_dataset.py <path_to_comma2k19_folder> --chunk_range <first_chunk> <last_chunk>
```
This script groups all data into folders corresponding to routes instead of having chunks and segments. For each route, it saves all image frames as jpgs and saves numpy arrays for each route corresponding to global pose frame times, frame positions, frame velocities, and frame orientations as well as synced CAN bus speeds and steering angles. The CAN bus data is to be used as labels for end-to-end control and the global pose data is to be used to generate paths for end-to-end planning labels.

## Generate train/val/test lists
Specify the routes to be used for the test set in data/dataset_lists/test_set_routes.json. Then, run the split_dataset script to generate the file splits for the training and validation sets:
```bash
python data/split_dataset.py <path_to_preprocessed_dataset> -f <number_of_future_steps> -p <number_of_past_steps> -d <dataset_size> -m <max_bin_size> -b <number_of_bins> -s <trainval_split>
```
This script will filter out paths where the car goes backwards and has an average speed of less than 10 m/s. Then it calculates a curvature score for the path based on the steering angle throughout the path. The final dataset is sampled in such a way that the dataset is more balanced based on this curvature score.

## Tasks
- [x] Preprocess dataset script
- [x] Generate train/val/test lists
- [ ] PyTorch Dataset class setup
- [ ] Transforms
- [ ] Simple CNN model
- [ ] Create training script
- [ ] Setup W&B or Tensorboard
- [ ] LSTM+CNN model
