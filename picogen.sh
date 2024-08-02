#!/usr/bin/env bash

cd $(dirname $0)

# Initialize variables
input_url_or_file=""
output_dir=""
ckpt_file=""
config_file=""
vocab_file=""
leadsheet_dir=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input_url_or_file)
      input_url_or_file="$2"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    --ckpt_file)
      ckpt_file="$2"
      shift 2
      ;;
    --config_file)
      config_file="$2"
      shift 2
      ;;
    --vocab_file)
      vocab_file="$2"
      shift 2
      ;;
    --leadsheet_dir)
      leadsheet_dir="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$leadsheet_dir" ]]; then
  leadsheet_dir=$output_dir/leadsheet
fi

# Check if required arguments are provided
if [[ -z "$input_url_or_file" || -z "$output_dir" || -z "$ckpt_file" || -z "$config_file" || -z "$vocab_file" ]]; then
  echo "Error: --input_url_or_file, --output_dir, --ckpt_file, --config_file, and --vocab_file are required."
  exit 1
fi

# Use the parsed arguments
echo "Input URL or File: $input_url_or_file"
echo "Output Directory: $output_dir"
echo "Checkpoint File: $ckpt_file"
echo "Config File: $config_file"
echo "Vocab File: $vocab_file"
echo "Lead Sheet Directory: $leadsheet_dir"

conda run -n sheetsage --no-capture-output pip install --upgrade yt-dlp > /dev/null 2>&1

# Stage1:
#     Extract the lead sheet from the input audio
# Arguments:
#     * input_url_or_file: the url or file path of the input audio
#     * output_dir: the directory to save the extracted lead sheet
conda run -n sheetsage --no-capture-output python -m picogen.infer stage1 \
  --input_url_or_file $input_url_or_file \
  --output_dir $output_dir \
 || exit 1

# Stage2:
#     Generate the piano cover from the extracted lead sheet
# Arguments:
#     * output_dir: the directory to find the extracted lead sheet and save the generated piano cover
#     * ckpt_file: the path to the checkpoint file
#     * config_file: the path to the config file
#     * vocab_file: the path to the vocab file

conda run -n picogen --no-capture-output python -m picogen.infer stage2 \
  --leadsheet_dir $leadsheet_dir \
  --output_dir $output_dir \
  --config_file $config_file \
  --vocab_file $vocab_file \
  --ckpt_file $ckpt_file \
