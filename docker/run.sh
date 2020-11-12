#!/bin/bash

version=$(cat $(dirname $(realpath -s $0))/"image_version.txt")

docker run -ti --rm --ipc=host --gpus all \
               -v /media/watouser/Seagate_Backup/datasets/comma2k19:/data/comma2k19 \
               -v /home/methier/end-to-end-driving:/home/repos/end-to-end-driving \
               end-to-end-driving:$version
