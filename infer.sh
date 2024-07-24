#!/usr/bin/env bash

cd $(dirname "$0")


input_url=""
input_audio=""
config_file=""
output_dir=""
ckpt_file=""
vocab_file=""
beat_file=""
sheetsage_file=""

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
        --config_file)
            config_file="$2"
            shift # past argument
            shift # past value
            ;;
        --output_dir)
            output_dir="$2"
            shift # past argument
            shift # past value
            ;;
        --ckpt_file)
            ckpt_file="$2"
            shift # past argument
            shift # past value
            ;;
        --vocab_file)
            vocab_file="$2"
            shift # past argument
            shift # past value
            ;;
        --beat_file)
            beat_file="$2"
            shift # past argument
            shift # past value
            ;;
        --sheetsage_file)
            sheetsage_file="$2"
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
echo "Config file: $config_file"
echo "Checkpoint file: $ckpt_file"
echo "Vocabulary file: $vocab_file"
if [ ! -z "$beat_file" ]; then
    echo "Beat file: $beat_file"
fi
if [ ! -z "$sheetsage_file" ]; then
    echo "SheetSage file: $sheetsage_file"
fi

if [ -z "$input_url" ] && [ -z "$input_audio" ]; then
    echo "Either input URL or input audio file must be provided."
    exit 1
fi

if [ -z $input_audio ]; then
    echo "Downloading input audio from $input_url"
    conda run -n picogen2 --no-capture-output \
        python -m picogen2 infer \
        --stage download \
        --input_url $input_url \
        --output_dir $output_dir
    input_audio=$output_dir/song.mp3
fi

if [ ! -f "$beat_file" ]; then
    echo "Extracting beat information from $input_audio"
    conda run -n picogen2 --no-capture-output \
        python -m picogen2 infer \
        --stage beat \
        --input_audio $input_audio \
        --output_dir $output_dir
fi

if [ ! -f "$sheetsage_file" ]; then
    echo "Extracting SheetSage features from $input_audio"
    conda run -n picogen2 --no-capture-output \
        python -m picogen2 infer \
        --stage sheetsage \
        --input_audio $input_audio \
        --output_dir $output_dir
fi

conda run -n picogen2 --no-capture-output \
    python -m picogen2 infer \
    --stage piano \
    --input_audio $input_audio \
    --output_dir $output_dir \
    --config_file $config_file \
    --ckpt_file $ckpt_file \
    --vocab_file $vocab_file
