#!/bin/bash
set -e

# Function to log messages
log_message() {
    printf "%s\n" "$1"
    printf "%s\n" "$1" >> "${SCRIPT_DIR}/setup_log.txt"
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

log_message "Script started at $(date)"

log_message "Creating Conda environment..."
conda env create -f environment.yml --quiet

log_message "Activating environment..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate beak-ta3

log_message "Installing local package..."
cd "${SCRIPT_DIR}/../.."
pip install -e .

log_message "Script finished at $(date)"
