#!/bin/bash

echo "Creating Docker container..."
cd "$(dirname "$0")/../.." || exit

docker run -it --rm --name beak-ta3 \
    -p 8888:8888 \
    -v $(pwd):/beak-ta3 \
    beak-ta3:latest \
    /bin/bash