
# Introduction

The environment can be set up
1. by following the descriptions in the [Readme](../../../README.md) provided on this repository or (**Docker** or **Conda**)
2. by importing the ready-to-go Docker container provided with this release

In any case, please consider to use either the **Docker install** or the **prepared container** over the conda install.

# Setup the pre-configured Docker container
## Installation

In case you select **option 2**, ensure to have the latest Docker version running on your system and follow the instructions below.

1. Create a terminal or command line instance and navigate to the **Docker** folder that has been created by extracting the provided data package (**zip** file)

2. Install the Docker **image** that contains the preconfigure environment

For **Linux or Mac**

```
cat beak_ta3_docker_image.tar | docker import - beak-ta3:latest
```
   
For **Windows**

```
type beak_ta3_docker_image.tar | docker import - beak-ta3:latest
```
   
3. Initialize/build a container

To be able to access the data, we need to connect a folder from our local system into the container: **replace** the `PATH_TO_BEAK_TA3_GITHUB_FOLDER` in the command below with the path of the `beak-ta3` folder that has been provided with the release.

```
docker run -i -t -p 8888:8888 -v PATH_TO_BEAK_TA3_GITHUB_FOLDER:/beak-ta3 --name beak-ta3 beak-ta3:latest /bin/bash
```

**Example**

```
docker run -i -t -p 8888:8888 -v C:/CMAAS/Docker_Example_Run_2024_08/GitHub/beak-ta3:/beak-ta3 --name beak-ta3 beak-ta3:latest /bin/bash
```

## Running the container and tools

### Terminal

1. Open a terminal instance
2. Start and attach the container: `docker start -ai beak-ta3` in the command line
3. Activate the environment: `conda activate beak-ta3`
4. Start Jupyter Lab: `jupyter lab --ip=0.0.0.0 --no-browser --allow-root`
5. Paste the provided URL (the one with the **IP**) into your webbrowser
### Docker Desktop
(GUI)

1. Open Docker Desktop
2. Start the container
3. Open a terminal instance
4. Attach the running container: `docker attach beak-ta3`
5. Activate the environment: `conda activate beak-ta3`
6. Start Jupyter Lab: `jupyter lab --ip=0.0.0.0 --no-browser --allow-root`
7. Paste the provided URL (the one with the **IP**) into your webbrowser

Instead of using Jupyter Lab, we recomment **VS-CODE**  that also supports notebooks and is easier to use in terms of activating the environment and navigation:

1. Open Docker Desktop
2. Start the container
3. Start VS-Code (**Dev containers** extension required)
4. Attach the container by selecting the `><` and choosing `Attach to Running Container` in the appearing drop-down list
5. Open a notebook and choose the `beak-ta3` Python interpreter when required
