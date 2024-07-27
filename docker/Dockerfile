FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG HOME="/root"
ARG APP_PATH=${HOME}/picogen2
WORKDIR ${APP_PATH}
SHELL ["/bin/bash", "-c"]

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential libopenmpi-dev \
    && apt-get install -y wget git \
    && apt-get install -y rubberband-cli fluidsynth ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n picogen2 python=3.10 -y && \
    conda install -n picogen2 -y mpi4py Cython && \
    conda run -n picogen2 --no-capture-output pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN git clone -b v2 https://github.com/tanchihpin0517/PiCoGen.git $APP_PATH
RUN conda run -n picogen2 --no-capture-output pip install -e .
