import os
from beak.integration.statmagic.call_som import run_som
from beak.experimental.io import create_file_list, check_path
from beak.utilities.helper import get_timestamp


# User inputs
MODEL_SETTINGS = "config_som.json"
DATA_FOLDER = "Path_to_data_folder"
LABELS = "Path_to_labels.tif"

# Use the current folder as working directory: results will be saved here in respective subfolders
ROOT_PATH_OUT = os.getcwd()

# Create model run output folder
MODEL_ID = get_timestamp()
OUT_PATH = os.path.join(ROOT_PATH_OUT, "models", "bnn", MODEL_ID)
check_path(OUT_PATH)

# Input files
file_list = create_file_list(DATA_FOLDER, out_string=True)

# Run BNN
output_prospectivity_layers = run_som(
    input_layers=file_list,
    input_labels=LABELS,
    config_file=MODEL_SETTINGS,
    output_folder=OUT_PATH,
)
