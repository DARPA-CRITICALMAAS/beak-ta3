#!/bin/bash

echo "Building Docker image..."
cd "$(dirname "$0")/../.." || exit

docker pull python:3.10-slim
docker build --no-cache --progress plain -t beak-ta3:latest -f setup/docker/Dockerfile .
