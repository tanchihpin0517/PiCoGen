#!/usr/bin/env bash

conda run -n picogen2 --no-capture-output pip install --upgrade yt-dlp >> /dev/null 2>&1

song_name="never_gonna_give_you_up"
conda run -n picogen2 --no-capture-output ./infer.sh \
    --input_url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' \
    --config_file ./assets/config.json \
    --output_dir ./demo_output/$song_name \
    --vocab_file ./assets/vocab.json \
    --beat_file ./demo_output/$song_name/song_beat.json \
    --sheetsage_file ./demo_output/$song_name/song_sheetsage.npz