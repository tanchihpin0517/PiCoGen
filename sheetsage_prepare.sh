#!/usr/bin/env bash

cd $(dirname $0)

python -m sheetsage.sheetsage.assets SHEETSAGE_V02_HANDCRAFTED ${JUKEBOX_CMD} && \
  python -m sheetsage.sheetsage.assets SHEETSAGE_V02_JUKEBOX && \
  python -m sheetsage.sheetsage.assets JUKEBOX
