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
COPY environment.yml /tmp
COPY requirements.txt /tmp

# Set temporary working directory
WORKDIR /tmp

# Update conda
RUN conda update -n base -c defaults conda

# Create a Conda environment with specified name
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "beak-ta3", "/bin/bash", "-c"]

# Set new environment to default
ENV CONDA_DEFAULT_ENV=beak-ta3

# Update pip and install additional requirements
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install -r requirements.txt

# Clean caches and temporary files
RUN conda clean --all -y
RUN pip cache purge	
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /beak-ta3