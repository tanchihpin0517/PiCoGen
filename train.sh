#!/usr/bin/env bash

cd $(dirname "$0")

# Function to display usage
function usage() {
    echo "Usage: $0 [command] [option...]"
    echo "Commands:"
    echo "download    Download Pop2Piano dataset from YouTube"
    echo "preprocess  Prepare necessary data for training"
    echo "train       Train the model"
}

command=$1
shift

case $command in
    "download")
        conda run -n picogen2 --no-capture-output \
            python -m picogen2 download \
            --song_file ./assets/training_dataset.csv \
            --data_dir ./data/dataset \
            $@
        ;;
    "preprocess")
        conda run -n picogen2 --no-capture-output \
            python -m picogen2 preprocess \
            --data_dir ./data/dataset \
            --output_dir ./data/processed \
            $@
        ;;
    *)
        echo "Invalid command: \"$command\""
        usage
        ;;
esac

