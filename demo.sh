#!/usr/bin/env bash

conda run -n picogen2 --no-capture-output pip install --upgrade yt-dlp >> /dev/null 2>&1

song_name="never_gonna_give_you_up"
conda run -n picogen2 --no-capture-output ./infer.sh \
    --input_url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' \
    --output_dir ./demo_output/$song_name