import warnings
import pandas as pd

from pathlib import Path
from beartype.typing import List, Tuple, Dict, Union

import beak.methods.som.argsSOM as asom
import beak.methods.som.do_nextsomcore_save_results as dnsr
from beak.hmi_integration.utils import create_file_list, _filter_files, create_zip_from_files
from cdr_schemas.prospectivity_input import ProspectivityOutputLayer

def run_som(
    input_layers: List[str],
    config_file: Path,
    output_folder: Path,
) -> List[Tuple[str, Dict]]:
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
    payload = json_data.loc["payload"]
    cma_id = payload["cma_id"]
    model_run_id = payload["model_run_id"]
    train_config = payload["event"]["train_config"]

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

    out_layers = _collect_results(
        cma_id=cma_id,
        model_run_id=model_run_id,
        kmeans=args.kmeans,
        output_folder=output_folder
    )

    # Remaining tasks
    # TODO: Label correlation
    # TODO: Plots

    return out_layers


def _collect_results(
    cma_id: str,
    model_run_id: str,
    kmeans: bool,
    output_folder: Union[str, Path]
) -> List[Tuple[str, Dict]]:
    """
    Creates information for CDR ProspectivityOutputLayer
    """
    init_meta = {
        "system": "statmagic",
        "system_version": "",
        "model": "SOM",
        "model_version":"0.0.1",
        "model_run_id": model_run_id,
        "output_type": "",
        "cma_id": cma_id,
        "title": "",
    }

    # Results with individual types and titles
    results = {
        "bmu_id": {
            "output_type": "cluster",
            "title": "Best Matching Units"
        },
        "q_error": {
            "output_type": "uncertainty",
            "title": "Quantization Error"
        },
        "cluster": {
            "output_type": "cluster",
            "title": "KMeans Clusters based on Best Matching Units"
        },
        "bmu_bmu_label_count": {
            "output_type": "cluster",
            "title": "Number of Labels per Best Matching Unit"
        },
        "bmu_cluster_label_count": {
            "output_type": "cluster",
            "title": "Number of Labels per Best Matching Unit grouped by KMeans-Cluster"
        }
    }

    # Create file lists and modify for kmeans since cluster.tif is created in any case
    results_file_list = create_file_list(
        folder=output_folder / "raster"
    )

    results_file_list = [file for file in results_file_list if not (
        "cluster.tif" in file and kmeans is False
    )]

    cdbk_vector_maps_file_list = _filter_files(
        file_list=results_file_list,
        file_suffix=None,
        file_prefix="b_"
    )

    plots_file_list = create_file_list(
        folder=output_folder / "plots",
        file_suffix=".png"
    )

    # Initialize output
    layers_list = []

    # Add raster results
    for file in results_file_list:
        file_stem = str(
            Path(file).stem
        ).lower()

        meta = init_meta.copy()

        for key, value in results.items():
            if key == file_stem:
                meta.update(value)

                layers_list.append(
                    (file, meta)
                )

    # Add codebook maps
    for file in cdbk_vector_maps_file_list:
        meta = init_meta.copy()
        meta.update(
            {
                "output_type": "codebook_map",
                "title": "Codebook Map"
            }
        )

        layers_list.append(
            (file, meta)
        )

    # Add plots
    if plots_file_list:
        plots_archive_path = str(output_folder / "plots.zip")

        create_zip_from_files(
            file_list=plots_file_list,
            archive_path=plots_archive_path
        )

        meta = init_meta.copy()
        meta.update(
            {
                "output_type": "plots",
                "title": "Archive containing generated Plots (Boxplots, Codebook Maps, Cluster Maps, Error Maps, ...)"
            }
        )

        layers_list.append(
            (plots_archive_path, meta)
        )
        
    prospectivity_output_layers = [] 
    for layer in layers_list:
        layer_path = layer[0]
        layer_meta = layer[1]
        
        layer_object = ProspectivityOutputLayer(**layer_meta)
        prospectivity_output_layers.append(
            (layer_path, layer_object)
        )
    
    return prospectivity_output_layers
