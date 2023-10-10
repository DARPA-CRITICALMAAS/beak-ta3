FROM ubuntu:22.04

EXPOSE 8888
EXPOSE 8000

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Use an official Miniconda3 image as a parent image
FROM continuumio/miniconda3:latest

# Make Docker use bash instead of sh
SHELL ["/bin/bash", "--login", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    python3-pip

# Copy the environment files into the container
COPY dependencies.yml /tmp/

# Update conda
RUN conda update -n base -c defaults conda

# Create a Conda environment with specified name
RUN conda env create -f /tmp/dependencies.yml

# Set new environment to default
ENV CONDA_DEFAULT_ENV=dev-env

# Clean conda cache
RUN conda clean --all -y	
	
# Set the working directory to /DIR
WORKDIR /methods

# Copy the current directory contents into the containers WORKDIR
COPY som /tmp/





