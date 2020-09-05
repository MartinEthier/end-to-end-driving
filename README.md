# end-to-end-driving

Repository to train end-to-end neural nets on the [comma2k19 dataset](https://github.com/commaai/comma2k19).

## Environment setup
Setup a conda environment with the libraries specified in requirements.txt

## Dataset setup
1. Download the dataset from [here](https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb) using the wget command and then run a command-line torrent client on the downloaded file to download the full dataset (I used rTorrent).
2. Run the pre-processing script as follows. Chunk range is set to 3-10 by default (Civic). Set to 1-2 for Rav4.
```bash
python process_dataset.py <path_to_comma2k19_folder> --chunk_range <first_chunk> <last_chunk>
```
This script groups all data into folders corresponding to routes instead of having chunks and segments. For each route, it saves all image frames as jpgs and saves numpy arrays for each route corresponding to global pose frame times, frame positions, frame velocities, and frame orientations as well as synced CAN bus speeds and steering angles. The CAN bus data is to be used as labels for end-to-end control and the global pose data is to be used to generate paths for end-to-end planning labels.
