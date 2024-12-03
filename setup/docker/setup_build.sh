#!/bin/bash

echo "Building Docker image..."
docker build -t beak-ta3:dev -f ../../setup/docker/Dockerfile ../../

echo "Running Docker container..."
docker run -it --name beak-ta3 \
    -p 8888:8888 \
    -v $(pwd)/../../:/beak-ta3/ \
    beak-ta3:dev
