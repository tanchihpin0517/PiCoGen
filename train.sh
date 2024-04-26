#!/usr/bin/env bash

cd $(dirname $0)

conda run -n picogen --no-capture-output python -m picogen.train \
  --config ./config/default.json \
  --vocab_file ./vocab.json \
  --checkpoint_path ./ckpt/default \
  --data_dir ./data/pop1k7 \
  $@
