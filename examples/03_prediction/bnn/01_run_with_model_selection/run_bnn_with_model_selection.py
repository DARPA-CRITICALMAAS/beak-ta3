# Imports
import os
import datetime

from beak.utilities.helper import get_timestamp

from beak.integration.statmagic.utils import (
    _get_data_folder,
)

from beak.integration.statmagic.call_bnn import run_bnn
from beak.experimental.io import load_model

# User inputs
from beak.models.hack_9m_poco import national_scale as model_config

MODEL_CONFIG = model_config["FEATURE_FOX"]
CONFIG_FILE = "bnn_config.json"
OUT_PATH = "Path_to_output_folder"

DATA_FOLDERS = [
    "Path_to_data_folder_1",
    "Path_to_data_folder_2",
]
LABELS = "Path_to_labels.tif"

MODEL_ID = get_timestamp()
OUT_PATH = os.path.join(OUT_PATH, "models", "bnn", MODEL_ID)

# Load model config
model_dict, file_list, counts = load_model(
    model=MODEL_CONFIG,
    folders=DATA_FOLDERS,
    verbose=0,
)

# Run BNN
output_prospectivity_layers = run_bnn(
    input_layers=file_list,
    input_labels=LABELS,
    config_file=CONFIG_FILE,
    output_folder=OUT_PATH,
)
