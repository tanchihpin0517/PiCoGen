FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG HOME="/home/picogen2"
ARG APP_PATH="$HOME/picogen2"
ARG DOCKER_USER="picogen2"

SHELL ["/bin/bash", "-c"]

# Install base utilities
RUN apt-get update \
    && apt install -y sudo gosu \
    && apt-get install -y build-essential libopenmpi-dev \
    && apt-get install -y wget git \
    && apt-get install -y rubberband-cli fluidsynth ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash $DOCKER_USER
RUN usermod -aG sudo $DOCKER_USER
USER $DOCKER_USER
RUN touch ~/.sudo_as_admin_successful
WORKDIR ${APP_PATH}

# Install miniconda
ENV CONDA_DIR=/home/picogen2/miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n picogen2 python=3.10 -y && \
    conda install -n picogen2 -y mpi4py Cython && \
    conda run -n picogen2 --no-capture-output pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN git clone -b v2 https://github.com/tanchihpin0517/PiCoGen.git $APP_PATH
RUN conda run -n picogen2 --no-capture-output pip install -e .

ENTRYPOINT [ "bash", "-c" ]
