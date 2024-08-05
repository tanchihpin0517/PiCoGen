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

# Check if the output directory is provided and exists
[ -z "$output_dir" ] && echo "Output directory must be provided." && exit 1
if [ ! -d "$output_dir" ]; then
    read -p "Output directory '$output_dir' does not exist. Create it? (Y/n): " create_dir
    create_dir=${create_dir,,} # convert input to lowercase
    if [[ $create_dir == "y" || $create_dir == "" ]]; then
        mkdir $output_dir || exit 1
    else
        echo "Directory not created. Exiting."
        exit 1
    fi
fi

function run {
    $@ || exit 1
}

if [ -z $docker_image ]; then
    # If input URL is provided, download the audio
    if [ -z "$input_audio" ]; then
        echo "Downloading input audio from $input_url"
        run pip install --upgrade yt-dlp > /dev/null 2>&1
        run python -m picogen2 infer \
            --stage download \
            --input_url $input_url \
            --output_dir $output_dir
        input_audio=$output_dir/song.mp3
    fi

    # Detect beats
    echo "Extracting beat information from $input_audio ..."
    run python -m picogen2 infer \
        --stage beat \
        --input_audio $input_audio \
        --output_dir $output_dir

    # Extract features
    echo "Extracting SheetSage features from $input_audio ..."
    run python -m picogen2 infer \
        --stage sheetsage \
        --input_audio $input_audio \
        --output_dir $output_dir

    # Infer piano cover
    echo "Infering piano cover ..."
    run python -m picogen2 infer \
        --stage piano \
        --input_audio $input_audio \
        --output_dir $output_dir

    exit 0
fi

# >>>>>>>>>>>>>>>>>>>>>> Docker <<<<<<<<<<<<<<<<<<<<<<<< #

docker_name="picogen2"
docker_cmd="docker exec -it $docker_name conda run -n picogen2 --no-capture-output"
docker_output_dir="/home/picogen2/docker_output"

echo "Run inference with Docker image: $docker_image"

# Initialize Docker
if docker ps -q --filter "name=$docker_name" | grep -q .; then
    echo "Container \`picogen2\` is already running"
else
    docker run --runtime=nvidia --gpus all -it -d --name $docker_name $docker_image bash
fi

function docker_run {
    run $docker_cmd $@
}

docker_run mkdir -p $docker_output_dir

# If input URL is provided, download the audio
if [ -z "$input_audio" ]; then
    echo "Downloading input audio from $input_url"
    docker_run pip install --upgrade yt-dlp > /dev/null 2>&1
    docker_run python -m picogen2 infer \
        --stage download \
        --input_url $input_url \
        --output_dir $docker_output_dir
    docker cp -q $docker_name:$docker_output_dir/song.mp3 $output_dir/song.mp3
else # copy the audio to the Docker container
    docker cp -q $input_audio $docker_name:$docker_output_dir/song.mp3
fi

input_audio=$docker_output_dir/song.mp3

# Detect beats
echo "Extracting beat information from $input_audio ..."
docker_run python -m picogen2 infer \
    --stage beat \
    --input_audio $input_audio \
    --output_dir $docker_output_dir
docker cp -q $docker_name:$docker_output_dir/song_beat.json $output_dir/song_beat.json

# Extract features
echo "Extracting SheetSage features from $input_audio ..."
docker_run python -m picogen2 infer \
    --stage sheetsage \
    --input_audio $input_audio \
    --output_dir $docker_output_dir
docker cp -q $docker_name:$docker_output_dir/song_sheetsage.npz $output_dir/song_sheetsage.npz

# Infer piano cover
echo "Infering piano cover ..."
docker_run python -m picogen2 infer \
    --stage piano \
    --input_audio $input_audio \
    --output_dir $docker_output_dir
docker cp -q $docker_name:$docker_output_dir/piano.mid $output_dir/piano.mid
docker cp -q $docker_name:$docker_output_dir/piano.txt $output_dir/piano.txt

# Clean up
if [ ! -z "$docker_image" ]; then
    echo "Stopping and removing Docker container \`$docker_name\`"
    docker stop $docker_name > /dev/null
    docker rm $docker_name > /dev/null
fi
