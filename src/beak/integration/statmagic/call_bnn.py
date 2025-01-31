import os

# Remove Tensorflow's warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from beak.utilities.sampling import (
    select_negative_samples,
    select_train_and_test_data,
    random_oversampling,
    extract_train_test_locations
)

from beak.utilities.conversion import convert_dtypes, expand_dims
from beak.utilities.file_io import (
    load_layers,
    load_layer,
    prepare_model_data,
    prepare_output,
    save_raster,
    write_json,
)

from beak.utilities.raster_processing import add_coordinates_to_raster
from beak.methods.bnn.fastBNN import fit_model, predict_model
from beak.methods.bnn.utils import build_network_architecture
from beak.evaluation.calculate_metrics import binary_classification

from beak.integration.statmagic.utils import (
    create_file_list,
    prepare_output_layers,
    export_train_test_splits,
    _create_model_run_config,
    _create_runtime_stats,
    _add_raster_results_to_output_layers,
    _create_prospectivity_output_layers
)

from cdr_schemas.prospectivity_input import ProspectivityOutputLayer


def run_bnn(
    input_layers: List[str],
    input_labels: str,
    config_file: str,
    output_folder: str,
    random_seed: int = 42,
) -> List[Tuple[str, Dict]]:
    """
    Call the BNN algorithm on input layers using the provided configuration.

    Args:
        input_layers: List containing the path of input rasters as strings.
        input_labels: String containing the path of the input labels.
        config_file: Path to the JSON configuration file.
        output_folder: Output folder for the results.
        random_seed: Random seed for reproducibility.

    Returns:
        None
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Prepare output folders
    raster_folder = os.path.join(output_folder, "raster")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(raster_folder, exist_ok=True)

    # Set output names for meta, splits and results
    metrics_name = "metrics"
    settings_name = "settings"
    runtime_name = "runtime"
    train_split_name = "train_split"
    test_split_name = "test_split"
    prediction_name = "likelihoods"
    uncertainty_name = "uncertainties"

    # Read model_run config
    json_data = pd.read_json(config_file)
    payload = json_data.loc["payload"]
    cma_id = payload["cma_id"]
    model_run_id = payload["model_run_id"]
    train_config = payload["event"]["train_config"]

    # Parameters
    fraction_train = train_config["train_size"]
    labels_init_negatives = train_config["init_negatives_multiplier"]
    labels_fraction_positives = train_config["upsample_positives_multiplier"]
    train_learning_rate = train_config["learning_rate"]
    train_epochs = train_config["training_epochs"]
    network_depth = train_config["network_arch_depth"]
    network_width = train_config["network_arch_width"]
    network_core = train_config["network_arch_core_units"]
    network_head = train_config["network_arch_head_units"]

    # Create model topology
    core_units, head_units = build_network_architecture(
        depth_per_network=network_depth,
        minimum_width=network_width,
        core_units = network_core if network_core else None,
        head_units = network_head if network_head else None,
    )

    # Runtime init
    start_time = datetime.now()

    # Load layers
    print("Load layers...")
    layers = load_layers(input_layers)
    labels, meta = load_layer(input_labels)

    # Add coordinates to raster
    print("Prepare model data...")
    layers_with_coords = add_coordinates_to_raster(
        src_array=layers,
        coord_file=input_labels
    )

    # Load initial data into array
    X_prepared, y_prepared = prepare_model_data(layers_with_coords, labels)
    assert np.any(y_prepared == 1), "No positive samples found. Please check your input data and consider imputing missing values."

    # Select random negatives
    print("Sample negatives...")
    X_sampled, y_sampled = select_negative_samples(
        X=X_prepared,
        y=y_prepared,
        multiplier=labels_init_negatives,
        random_seed=random_seed,
    )

    # Create train and test sets
    print("Create splits...")
    X_train_split, X_test_split, y_train_split, y_test_split = select_train_and_test_data(
        X=X_sampled,
        y=y_sampled,
        train_size=fraction_train,
        random_seed=random_seed,
    )

    # Create train and test split outputs and locations
    X_train_split, train_locations = extract_train_test_locations(X_train_split, y_train_split, crs=meta["crs"])
    X_test_split, test_locations = extract_train_test_locations(X_test_split, y_test_split, crs=meta["crs"])

    # Oversample positives
    print("Modify labels...")
    X_train, y_train = random_oversampling(
        X=X_train_split,
        y=y_train_split,
        sampling_strategy=labels_fraction_positives,
        random_seed=random_seed,
    )

    # Extend shapes and cast dtype
    X_train, X_test, y_train, y_test = convert_dtypes([X_train, X_test_split, y_train, y_test_split])
    y_train, y_test = expand_dims([y_train, y_test])

    # Create model inputs
    buffer_size = tf.data.experimental.UNKNOWN_CARDINALITY
    data_train = tf.data.Dataset.from_tensor_slices(
        (
            X_train,
            y_train,
        )
    ).shuffle(buffer_size).batch(512)

    # Run training
    print("\nTraining...")
    feature_count = X_train.shape[1]
    data_count = X_train.shape[0]

    model, loss = fit_model(
        data_train=data_train,
        input_shape=feature_count,
        core_units=core_units,
        head_units=head_units,
        lr=train_learning_rate,
        epochs=train_epochs,
        N=data_count,
    )

    # Evaluation
    print("\nEvaluation...")
    metrics_train = binary_classification(model, X_train_split, y_train_split)
    evaluations =  {"train": metrics_train}

    if y_test is not None:
        metrics_test = binary_classification(model, X_test, y_test)
        evaluations.update({"test": metrics_test})

    # Prediction
    print("Prediction...")
    prediction, uncertainty = predict_model(
        model,
        layers,
        target_shape=(meta["height"], meta["width"])
    )

    # Save rasters
    target_meta = meta.copy()
    target_meta.update(
        {
            "nodata": -99999
        }
    )

    results_data = (prediction, uncertainty)
    results_files = (prediction_name, uncertainty_name)

    for result, name in zip(results_data, results_files):
        out_file = os.path.join(raster_folder, name + ".tif")
        out_array, out_meta = prepare_output(result, target_meta)
        save_raster(out_array, out_meta, out_file)

    # Extract values from points for training and testing datasets
    export_train_test_splits(
        src_data=(
            [prediction, uncertainty],
            [prediction_name, uncertainty_name]
        ),
        split_data=(
            [train_locations, test_locations],
            [train_split_name, test_split_name]
        ),
        template=(labels, meta),
        output_folder=output_folder,
    )

    # Runtime final
    end_time = datetime.now()
    duration = end_time - start_time
    runtime = duration.total_seconds()

    # Collect metadata
    print("Create and collect outputs...")
    network_arch = {
        "core_units": core_units,
        "head_units": head_units,
    }

    label_data = [
        ("initial", _get_label_info(y_prepared)),
        ("sampled_negatives", _get_label_info(y_sampled)),
        ("train_split", _get_label_info(y_train_split)),
        ("test_split", _get_label_info(y_test_split)),
        ("final", _get_label_info(y_train))
    ]

    additional_config_info = {
        "network_architecture": network_arch,
        "sampled_labels": _get_sampled_labels(label_data)
    }

    model_config = _create_model_run_config(
        model_run=(cma_id, model_run_id),
        input_data=(input_layers, input_labels),
        train_config=train_config,
        **additional_config_info
    )

    additional_runtime_info = {
        "input_labels": sum( _get_label_info(y_train).values()),
        "nn_epochs": train_epochs,
        "nn_parameters": _get_network_parameters(feature_count, core_units + head_units + [1])
    }

    runtime_stats = _create_runtime_stats(
        input_size=feature_count,
        meta=meta,
        runtime=runtime,
        **additional_runtime_info
    )

    # Set outputs
    output_locations = (
        output_folder,
        raster_folder
    )

    output_files = (
        metrics_name,
        settings_name,
        train_split_name,
        test_split_name,
        runtime_name
    )

    # Save metrics and model metadata
    write_json(output_folder, metrics_name + ".json", evaluations)
    write_json(output_folder, settings_name + ".json", model_config)
    write_json(output_folder, runtime_name + ".json", runtime_stats)

    # Collect results
    out_layers = _collect_results(
        cma_id=cma_id,
        model_run_id=model_run_id,
        output_locations=output_locations,
        output_files=output_files
    )

    return out_layers


def _get_label_info(
    data: Optional[np.ndarray],
) -> Dict:
    """
    Get label information for positives and negatives.

    Args:
        data: Input data to count positives and negatives.

    Returns:
        dict: Dictionary containing count of positives and negatives.
    """
    if data is not None:
        positives = np.count_nonzero(data == 1)
        negatives = np.count_nonzero(data == 0)
        custom_negatives = np.count_nonzero(data == -1)

        out_dict = {
            "positives": positives,
            "negatives": custom_negatives if custom_negatives > 0 else negatives
        }
    else:
        out_dict = {}

    return out_dict


def _get_sampled_labels(
    label_data: List[Tuple[str, Dict]],
):
    """
    Create a dictionary containing all settings.

    Args:
        label_data: Data and descriptions for labels.

    Returns:
        dict: Dictionary containing labels processed in the workflow.
    """
    labels = {}
    for description, data in label_data:
        labels.update(
            {
                description: data
            }
        )

    return labels


def _get_network_parameters(
    input_size: int,
    nodes: List[int],
) -> int:
    """
    Get the number of parameters in the neural network.

    Args:
        input_size: Size of the input layer.
        nodes: Number of neurons for each layer.
    """
    # Input to first hidden layer
    weight_params = input_size * nodes[0]

    # Hidden layers to output layer
    for i in range(1, len(nodes)):
        weight_params += nodes[i - 1] * nodes[i]

    # Add bias parameters for each layer
    bias_params = sum(nodes)

    return weight_params + bias_params


def _collect_results(
    cma_id: str,
    model_run_id: str,
    output_locations: Tuple[str, ...],
    output_files: Tuple[str,...],
) -> List[Tuple[str, ProspectivityOutputLayer]]:
    """
    Create information for the CDR ProspectivityOutputLayer Class.

    Args:
        cma_id: Unique identifier for the CMA.
        model_run_id: Unique identifier for the model run.
        output_locations: The output folders and locations for metrics and meta files.

    Returns:
        List of tuples, where each tuple contains the file path and the related ProspectivityOutputLayer object.
    """
    output_folder, raster_folder = output_locations
    metrics, settings, train_split, test_split, runtime_stats = output_files

    # Initialization
    init_meta = {
        "system": "statmagic",
        "system_version": "",
        "model": "BNN",
        "model_version": "0.0.1",
        "model_run_id": model_run_id,
        "output_type": "",
        "cma_id": cma_id,
        "title": "",
    }

    # Results with individual types and titles
    results = {
        "likelihoods": {
            "output_type": "likelihood",
            "title": "Predicted probabilities"
        },
        "uncertainties": {
            "output_type": "uncertainty",
            "title": "Uncertainty of the prediction result"
        },
    }

    # Create file list for raster results
    results_file_list = create_file_list(
        folder=raster_folder
    )

    # Initialize output
    layers_list = _add_raster_results_to_output_layers(
        file_list=results_file_list,
        results=results,
        init_meta=init_meta
    )

    # Add metrics
    update_meta = {
        "output_type": "metrics",
        "title": "Archive containing the calculated metrics"
    }

    layers_list = prepare_output_layers(
        layers=layers_list,
        files=os.path.join(output_folder, metrics + ".json"),
        meta=(init_meta, update_meta),
        output=(output_folder, metrics + ".zip")
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

    # Add splits
    update_meta = {
        "output_type": "train_test_data",
        "title": "Archive containing the train and test data"
    }

    splits = [train_split, test_split]
    extensions = [".csv", ".gpkg"]

    model_data = [
        os.path.join(output_folder, split + extension)
        for split, extension in product(splits, extensions)
    ]

    layers_list = prepare_output_layers(
        layers=layers_list,
        files=model_data,
        meta=(init_meta, update_meta),
        output=(output_folder, "train_test_data.zip")
    )

    # Create CDR object
    prospectivity_output_layers = _create_prospectivity_output_layers(layers_list)

    return prospectivity_output_layers
