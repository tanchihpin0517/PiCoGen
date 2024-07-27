#!/usr/bin/env bash

cd $(dirname $0)/..

DOCKER_NAMESPACE=tanchihpin0517
DOCKER_NAME=picogen2
DOCKER_TAG=latest

mkdir -p ./docker_temp
rsync -a ~/.piano_transcription_inference_data/ ./docker_temp/.piano_transcription_inference_data/
rsync -a ~/.mirtoolkit/ ./docker_temp/.mirtoolkit/
rsync -a ~/.cache/picogen2/ ./docker_temp/picogen2/
rsync -a ~/.cache/jukebox/ ./docker_temp/jukebox/
rsync -a ~/.sheetsage/sheetsage/ ./docker_temp/sheetsage/

docker build -t ${DOCKER_NAMESPACE}/${DOCKER_NAME}:${DOCKER_TAG} -f docker/Dockerfile $@ .

docker build -t ${DOCKER_NAMESPACE}/${DOCKER_NAME}:${DOCKER_TAG}-full -f docker/Dockerfile-full $@ .
