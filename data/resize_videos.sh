#!/bin/bash
# Note: Need to have comma10k conda env activated to get ffmpeg
ROOT_DIR="$1"
SIZE="$2" # WxH -> 384x288
echo $ROOT_DIR

for chunk_dir in $ROOT_DIR/*; do
    for route_dir in $chunk_dir/*; do
        for split_dir in $route_dir/*; do
            ffmpeg -y -i $split_dir/"video.hevc" -s $SIZE -c:a copy $split_dir/"video_$SIZE.hevc"
        done
    done
done
    
