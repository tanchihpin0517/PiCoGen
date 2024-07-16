#!/usr/bin/env bash

cd $(dirname $0)

conda run -n picogen --no-capture-output \
  python -m picogen.repr gen_vocab \
  --output_file ./data/vocab.json

conda run -n picogen --no-capture-output \
  python -m picogen.data gen_cache \
  --data_dir ./data/pop1k7 \
  --vocab_file ./data/vocab.json \
  --config ./config/default.json
