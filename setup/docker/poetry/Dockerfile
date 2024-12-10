# Use an official Python runtime
FROM python:3.10-slim

# Set environment variables
ENV BASE_PATH=/beak-ta3

# Set work directory
WORKDIR $BASE_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml $BASE_PATH/
COPY README.md $BASE_PATH/
COPY src/ $BASE_PATH/src/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Install additional packages within the virtual environment
RUN poetry run pip install gdal==3.6.0 somoclu==1.7.6 --no-deps

# Install local package in editable mode
RUN poetry run pip install -e .

# Clear caches
RUN apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && rm -rf ~/.cache/pypoetry \
    && pip cache purge
