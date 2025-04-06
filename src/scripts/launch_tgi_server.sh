#!/bin/sh

systemctl --user start docker

DEVICE=$1
PORT=$2

# model=facebook/opt-125m
model=google/flan-t5-xxl
# model=stabilityai/StableBeluga2

# volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
volume=/local/scratch/$(whoami)/docker/data
token=$HF_READ_TOKEN
# token=$FOOBAR
echo $token

# ghcr.io/huggingface/text-generation-inference:1.3
echo docker run --gpus '"device='${DEVICE}'"' --shm-size 1g -p $PORT:80 -e HUGGING_FACE_HUB_TOKEN=$token -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model --max-batch-total-tokens 32000 --max-concurrent-requests 500
docker run --gpus '"device='${DEVICE}'"' --shm-size 1g -p $PORT:80 -e HUGGING_FACE_HUB_TOKEN=$token -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model --max-batch-total-tokens 32000 --max-concurrent-requests 500

# systemctl --user stop docker
