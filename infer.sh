#!/usr/bin/env bash

cd $(dirname "$0")

input_url=""
input_audio=""
output_dir=""
docker_image=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_url)
            input_url="$2"
            shift # past argument
            shift # past value
            ;;
        --input_audio)
            input_audio="$2"
            shift # past argument
            shift # past value
            ;;
        --output_dir)
            output_dir="$2"
            shift # past argument
            shift # past value
            ;;
        --docker_image)
            docker_image="$2"
            shift # past argument
            shift # past value
            ;;
        *)    # unknown option
            # If it's a command or another flag, break the loop
            shift
            ;;
    esac
done

echo "Input URL: $input_url"
echo "Input audio: $input_audio"
echo "Output directory: $output_dir"
echo "Docker image: $docker_image"

if [ -z "$input_url" ] && [ -z "$input_audio" ]; then
    echo "Either input URL or input audio file must be provided."
    exit 1
fi

if [ -z "$input_audio" ]; then
    echo "Downloading input audio from $input_url"
    conda run -n picogen2 --no-capture-output \
        python -m picogen2 infer \
        --stage download \
        --input_url $input_url \
        --output_dir $output_dir || exit 1

    input_audio=$output_dir/song.mp3
fi

echo "Extracting beat information from $input_audio"
conda run -n picogen2 --no-capture-output \
    python -m picogen2 infer \
    --stage beat \
    --input_audio $input_audio \
    --output_dir $output_dir || exit 1

echo "Extracting SheetSage features from $input_audio"
conda run -n picogen2 --no-capture-output \
    python -m picogen2 infer \
    --stage sheetsage \
    --input_audio $input_audio \
    --output_dir $output_dir || exit 1

conda run -n picogen2 --no-capture-output \
    python -m picogen2 infer \
    --stage piano \
    --input_audio $input_audio \
    --output_dir $output_dir