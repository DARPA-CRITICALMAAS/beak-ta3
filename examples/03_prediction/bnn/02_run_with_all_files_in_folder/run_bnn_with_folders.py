# Imports
import os
import datetime

from beak.integration.statmagic.utils import (
    _get_data_folder,
)

from beak.integration.statmagic.call_bnn import run_bnn
from beak.experimental.io import create_file_list, check_path
from beak.utilities.helper import get_timestamp

# User inputs
CONFIG_FILE = "bnn_config.json"
DATA_FOLDER = "Path_to_data_folder"
OUT_PATH = "Path_to_output_folder"
LABELS = "Path_to_labels.tif"

MODEL_ID = get_timestamp()
OUT_PATH = os.path.join(OUT_PATH, "models", "bnn", MODEL_ID)
check_path(OUT_PATH)

# Input files
file_list = create_file_list(DATA_FOLDER)
file_list = [str(file) for file in file_list]

# Run BNN
output_prospectivity_layers = run_bnn(
    input_layers=file_list,
    input_labels=LABELS,
    config_file=CONFIG_FILE,
    output_folder=OUT_PATH,
)
