#!/usr/bin/env bash

# Check if the first argument is "full"
if [ "$1" == "full" ]; then
    IMAGE_TAG="latest-full"
elif [ -z "$1" ]; then
    IMAGE_TAG="latest"
else
    echo "Error: Invalid argument. Only 'full' is accepted."
    exit 1
fi

docker run --rm --runtime=nvidia --gpus all \
    -it --entrypoint bash tanchihpin0517/picogen2:$IMAGE_TAG