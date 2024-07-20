#!/usr/bin/env bash

cd $(dirname "$0")

conda create -n picogen2 python=3.10 -y
conda install -n picogen2 mpi4py -y
conda run -n picogen2 --no-capture-output pip install -r requirements.txt

