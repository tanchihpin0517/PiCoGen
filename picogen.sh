#!/usr/bin/env bash

# conda run -n picogen --no-capture-output python -m picogen.infer \
#   --input_url_or_file https://www.youtube.com/watch?v=8SrtnehtGh0 \
#   --output_dir ./output/invu \
#   --leadsheet_dir ./output/leadsheet \
#   --config_file ./config/default.json \
#   --ckpt_file ./ckpt/default/models/model_00075000 \
#   --vocab_file ./vocab.json \
#   $@

conda run -n picogen --no-capture-output python -m picogen.infer \
  --config_file ./config/default.json \
  --ckpt_file ./ckpt/default/models/model_00075000 \
  --vocab_file ./asset/vocab.json \
  $@
