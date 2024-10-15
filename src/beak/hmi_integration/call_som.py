import warnings
import pandas as pd

from pathlib import Path
from beartype.typing import List

import beak.methods.som.argsSOM as asom
import beak.methods.som.do_nextsomcore_save_results as dnsr
from beak.hmi_integration.utils import _extract_payload


def run_som(
    input_layers: List[str],
    config_file: Path,
    output_folder: Path,
) -> None:
    """
    Calls the SOM algorithm on input layers using the provided configuration.

    Args:
        input_layers (List[str]): List of input geotiffs.
        config_file (Path): Path to the JSON configuration file.
        output_folder (Path): Output folder for the SOM results.

    Returns:
        None
    """
    # Initialize args
    args = asom.Args()

    # Prepare inputs
    input_file = ','.join(input_layers)
    args.input_file = input_file
    args.geotiff_input = input_file

    # Check output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define SOM outputs
    args.output_folder = str(output_folder)
    args.output_file_somspace = str(output_folder / "result_som.txt")
    args.output_file_geospace = str(output_folder / "result_geo.txt")
    args.outgeofile = args.output_file_geospace

    # Read config
    json_data = pd.read_json(config_file)
    train_config = _extract_payload(json_data, target="train_config", normalize=False)

    # Set SOM arguments
    args.initialcodebook = None
    args.som_x = train_config["dimensions_x"]
    args.som_y = train_config["dimensions_y"]
    args.epochs = train_config["num_epochs"]
    args.neighborhood = train_config["neighborhood_function"]
    args.std_coeff = train_config["gaussian_neighborhood_coefficient"]
    args.maptype = train_config["som_type"]
    args.radius0 = train_config["initial_neighborhood_size"]
    args.radiusN = train_config["final_neighborhood_size"]
    args.radiuscooling = train_config["neighborhood_decay"]
    args.scalecooling = train_config["learning_rate_decay"]
    args.scale0 = train_config["initial_learning_rate"]
    args.scaleN = train_config["final_learning_rate"]
    args.initialization = train_config["som_initialization"]
    args.gridtype = train_config["grid_type"]

    # Set k-means arguments
    args.kmeans = True
    args.kmeans_init = train_config["num_initializations"]
    args.kmeans_min = 20
    args.kmeans_max = 50

    # Set label arguments
    args.label = False

    # Run SOM without k-means warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dnsr.run_SOM(args)

    # Remaining tasks
    # TODO: Label correlation
    # TODO: Plots
