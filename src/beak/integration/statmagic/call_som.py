import os
import warnings
import pandas as pd
import rasterio

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import beak.methods.som.argsSOM as asom
import beak.methods.som.do_nextsomcore_save_results as dnsr
import beak.methods.som.argsPlot as aplot
import beak.methods.som.plot_som_results as plot

from beak.integration.statmagic.utils import (
    create_file_list,
    prepare_output_layers,
    delete_files,
    _filter_files,
    _filter_layers_from_payload,
    _create_model_run_config,
    _create_runtime_stats,
    _add_raster_results_to_output_layers,
    _create_prospectivity_output_layers
)

from beak.utilities.file_io import write_json
from cdr_schemas.prospectivity_input import ProspectivityOutputLayer


def run_som(
    input_layers: List[str],
    input_labels: Optional[str],
    config_file: str,
    output_folder: str,
) -> List[Tuple[str, Dict]]:
    """
    Call the SOM algorithm on input layers using the provided configuration.

    Args:
        input_layers: List containing the path of input rasters as strings.
        input_labels: String containing the path of the input labels.
        config_file: Path to the JSON configuration file.
        output_folder: Output folder for the SOM results.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    # Initialize arguments
    args = asom.Args()
    argsP = aplot.Args()

    # SOM inputs and outputs
    input_layer_string = ','.join(input_layers)
    args.input_file = input_layer_string
    args.geotiff_input = input_layer_string

    if input_labels is not None:
        args.label = True
        args.label_geotiff_file = input_labels
    else:
        args.label = False

    args.output_folder = output_folder
    args.output_file_somspace = os.path.join(output_folder, "result_som.txt")
    args.output_file_geospace = os.path.join(output_folder, "result_geo.txt")
    args.outgeofile = args.output_file_geospace
    config_name = "settings"
    runtime_name = "runtime"

    # Plot input and outputs
    argsP.input_file = args.input_file
    argsP.dir = args.output_folder
    argsP.outsomfile = args.output_file_somspace
    argsP.outgeofile = args.output_file_geospace

    # Pack output folders
    output_locations = (
        output_folder,
        os.path.join(output_folder, args.output_raster_folder),
        os.path.join(output_folder, argsP.output_plots_folder)
    )

    output_files = (
        config_name,
        runtime_name
    )

    # Read model_run config
    json_data = pd.read_json(config_file)
    payload = json_data.loc["payload"]
    cma_id = payload["cma_id"]
    model_run_id = payload["model_run_id"]
    train_config = payload["event"]["train_config"]

    # SOM arguments
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

    # K-means arguments
    args.kmeans = train_config["kmeans"]
    args.kmeans_min = train_config["kmeans_min"]
    args.kmeans_max = train_config["kmeans_max"]
    args.kmeans_init = train_config["num_initializations"]

    # Plot arguments
    argsP.som_x = args.som_x
    argsP.som_y = args.som_y
    argsP.grid_type = args.gridtype

    # Runtime init
    start_time = datetime.now()

    # Run SOM without k-means warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dnsr.run_SOM(args)

    # Plotting
    plot.run_plotting_script(argsP)

    _create_layer_plot_names(
        payload=payload,
        label_column="label_raster",
        output_folder=output_locations[2],
    )

    # Runtime final
    end_time = datetime.now()
    duration = end_time - start_time
    runtime = duration.total_seconds()

    # Collect metadata
    model_config = _create_model_run_config(
        model_run=(cma_id, model_run_id),
        input_data=(input_layers, input_labels),
        train_config=train_config,
    )

    additional_runtime_info = {
        "som_size": args.som_x * args.som_y,
        "som_epochs": args.epochs,
        "kmeans_clusters": args.kmeans_max
    }

    runtime_stats = _create_runtime_stats(
        input_size=len(input_layers),
        meta=rasterio.open(input_layers[0]).meta,
        runtime=runtime,
        **additional_runtime_info
    )

    # Save model settings and metadata
    write_json(output_folder, config_name + ".json", model_config)
    write_json(output_folder, runtime_name + ".json", runtime_stats)

    # Collect results
    out_layers = _collect_results(
        cma_id=cma_id,
        model_run_id=model_run_id,
        kmeans=args.kmeans,
        output_locations=output_locations,
        output_files=output_files
    )

    # Delete intermediate files
    _delete_intermediate_files(output_folder)

    return out_layers


def _delete_intermediate_files(output_folder: str) -> None:
    """
    Delete intermediate files from the output folder.

    Args:
        output_folder: Output folder for the intermediate files.

    Returns:
        None
    """
    files = create_file_list(
        folder=output_folder,
        file_suffix=(".txt", ".dictionary"),
        file_prefix=None
    )

    if files:
        delete_files(files)


def _create_layer_plot_names(
    payload: pd.DataFrame,
    label_column: Optional[str],
    output_folder: str,
) -> None:
    """
    Create a table for layer names and titles with indices for each evidence layer.

    Args:
        payload: DataFrame containing the payload.
        label_column: Column name containing labels.
        output_folder: Output folder for the plot names.

    Returns:
        None
    """
    layers = _filter_layers_from_payload(
        payload=payload,
        label_column=label_column,
        filter_labels=False
    )

    if not layers.empty:
        layers = layers[
            ["title", "layer_id", "download_url"]
        ]
        layers.insert(0, "plot_index", range(1, len(layers) + 1))
        layers.insert(2, "file_name", layers["download_url"].apply(lambda x: Path(x).name))

        output_path = os.path.join(output_folder, "plot_names.csv")
        layers.to_csv(output_path, sep=";", index=False)


def _collect_results(
    cma_id: str,
    model_run_id: str,
    kmeans: bool,
    output_locations: Tuple[str, ...],
    output_files: Tuple[str, ...]
) -> List[Tuple[str, ProspectivityOutputLayer]]:
    """
    Create information for the CDR ProspectivityOutputLayer Class.

    Args:
        cma_id: Unique identifier for the CMA.
        model_run_id: Unique identifier for the model run.
        kmeans: Boolean indicating if k-means clustering was performed.
        output_locations: Tuple containing the output folder, raster folder, and plots folder.

    Returns:
        List of tuples, where each tuple contains the file path and the related ProspectivityOutputLayer object.
    """
    output_folder, raster_folder, plots_folder = output_locations
    settings, runtime_stats = output_files

    # Initialization
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
            "title": "K-Means Clusters"
        },
        "bmu_bmu_label_count": {
            "output_type": "cluster",
            "title": "Number of Labels per Best Matching Unit"
        },
        "bmu_cluster_label_count": {
            "output_type": "cluster",
            "title": "Number of Labels per Best Matching Unit grouped by K-Means-Clusters"
        }
    }

    # Create file lists and modify for kmeans since "cluster.tif" is created in any case
    results_file_list = create_file_list(
        folder=raster_folder
    )

    results_file_list = [file for file in results_file_list if not (
        "cluster.tif" in file and kmeans is False
    )]

    # Codebook map file list
    codebook_maps_file_list = _filter_files(
        file_list=results_file_list,
        file_suffix=None,
        file_prefix="b_"
    )

    # Plot files list including title-name correlation
    plots_file_list = create_file_list(
        folder=plots_folder,
        file_suffix=(".png", ".csv")
    )

    # Initialize output
    layers_list = _add_raster_results_to_output_layers(
        file_list=results_file_list,
        results=results,
        init_meta=init_meta
    )

    # Add codebook maps, names starting with "b_"
    for file in codebook_maps_file_list:
        file_name = str(
            Path(file).stem
        )[2:]

        update_meta = {
            "output_type": "codebook_map",
            "title": f"Codebook Map {file_name}"
        }

        layers_list = prepare_output_layers(
            layers=layers_list,
            files=file,
            meta=(init_meta, update_meta),
            output=None
        )

    # Add plots
    if plots_file_list:
        update_meta = {
            "output_type": "plots",
            "title": "Archive containing generated Plots (Boxplots, Codebook Maps, Cluster Maps, Error Maps, ...)"
        }

        layers_list = prepare_output_layers(
            layers=layers_list,
            files=plots_file_list,
            meta=(init_meta, update_meta),
            output=(output_folder, "plots.zip")
        )

    # Add metadata and settings
    update_meta = {
        "output_type": "metadata",
        "title": "Archive containing model metadata and settings"
    }

    layers_list = prepare_output_layers(
        layers=layers_list,
        files=os.path.join(output_folder, settings + ".json"),
        meta=(init_meta, update_meta),
        output=(output_folder, settings + ".zip")
    )

    # Add runtime stats
    update_meta = {
        "output_type": "runtime",
        "title": "Archive containing runtime statistics"
    }

    layers_list = prepare_output_layers(
        layers=layers_list,
        files=os.path.join(output_folder, runtime_stats + ".json"),
        meta=(init_meta, update_meta),
        output=(output_folder, runtime_stats + ".zip")
    )

    # Create CDR object
    prospectivity_output_layers = _create_prospectivity_output_layers(layers_list)

    return prospectivity_output_layers
