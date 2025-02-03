import os
from beak.experimental.io import create_model_setup
from beak.integration.statmagic.call_bnn import run_bnn
from beak.utilities.helper import get_runtime


# Prepare configs and data
train_config, file_list, labels, out_path = create_model_setup(
    method="bnn",
    work_dir=os.path.abspath(os.path.dirname(__file__))
)

# Run BNN
print("\nStarting BNN Run:\n")

output_prospectivity_layers = get_runtime(
    function=run_bnn,
    input_layers=file_list,
    input_labels=labels,
    config_file=train_config,
    output_folder=out_path,
)
