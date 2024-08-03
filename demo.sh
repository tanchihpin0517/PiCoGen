#!/usr/bin/env bash

cd $(dirname "$0")

song_name="never_gonna_give_you_up"
./infer.sh \
    --input_url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' \
    --output_dir ./demo_output/$song_name \
    $@