import os
from beak.utilities.helper import get_timestamp
from beak.integration.statmagic.utils import _get_data_folder
from beak.integration.statmagic.call_bnn import run_bnn
from beak.experimental.io import load_model


# User inputs, model config and files
from beak.models.hack_6m_mama_nico import national_scale as model_config            # Change for different CMAs

MODEL_CONFIG = model_config["BASELINE_BISON"]                                       # Change for other model config
MODEL_SETTINGS = "config_bnn.json"

# Results will be saved here in separate model-run subfolders
ROOT_PATH_OUT = os.getcwd()

# Root folder for data
DATA_ROOT = _get_data_folder() / "PROCESSED" / "national_us_cont_102008_2240"       # Change for different CMAs

# Used for loading the defined datasets from the model config
DATA_FOLDERS = [
    DATA_ROOT / "unified_scaled_std",                                               # Change for different CMAs
    DATA_ROOT / "unified_scaled_log",                                               # Add or remove folders
    DATA_ROOT / "unified_lawley/numerical_scaled_std"                               # Be specific
]

LABELS_NAME = "EPSG_102008_RES_2240_us_cont_MAMA_NICO_HM6_TA2_HYPERSITE.tif"        # Fill name for labels
LABELS = DATA_ROOT / "labels" / LABELS_NAME

# Load model config and files
model_dict, file_list, counts = load_model(
    model=MODEL_CONFIG,
    folders=DATA_FOLDERS,
    verbose=0,
)

# Create model run subfolder and run model
MODEL_ID = get_timestamp()
OUT_PATH = os.path.join(ROOT_PATH_OUT, "models", "bnn", MODEL_ID)

output_prospectivity_layers = run_bnn(
    input_layers=file_list,
    input_labels=LABELS,
    config_file=MODEL_SETTINGS,
    output_folder=OUT_PATH,
)
