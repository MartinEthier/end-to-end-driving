#!/bin/bash

version=$(cat $(dirname $(realpath -s $0))/"image_version.txt")

docker build --rm --tag end-to-end-driving:$version .