import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from beak.utilities.sampling import (
    select_negative_samples,
    select_train_and_test_data,
    random_oversampling,
)

from beak.utilities.conversion import convert_dtypes, expand_dims
from beak.utilities.file_io import (
    load_layers,
    load_layer,
    prepare_model_data,
    prepare_output,
    save_raster,
    write_json
)
from beak.methods.bnn.fastBNN import fit_model, predict_model
from beak.methods.bnn.utils import build_network_architecture
from beak.evaluation.calculate_metrics import binary_classification

from beak.integration.statmagic.utils import (
    create_file_list,
    prepare_output_layers,
)

from cdr_schemas.prospectivity_input import ProspectivityOutputLayer


# Remove Tensorflow's warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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

    Returns:
        None
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Prepare output folders
    raster_folder = os.path.join(output_folder, "raster")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(raster_folder, exist_ok=True)

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

    # Load layers
    input_layers = load_layers(input_layers)
    input_labels, meta = load_layer(input_labels)

    # Load initial data into array
    X_prepared, y_prepared = prepare_model_data(input_layers, input_labels)

    # Select random negatives
    X_sampled, y_sampled = select_negative_samples(
        X=X_prepared,
        y=y_prepared,
        multiplier=labels_init_negatives,
        random_seed=random_seed,
    )

    # Create train and test sets
    X_train_split, X_test_split, y_train_split, y_test_split = select_train_and_test_data(
        X=X_sampled,
        y=y_sampled,
        train_size=fraction_train,
        random_seed=random_seed,
    )

    # Oversample positives
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
    metrics_train = binary_classification(model, X_train_split, y_train_split)
    evaluations =  {"train": metrics_train}

    if y_test is not None:
        metrics_test = binary_classification(model, X_test, y_test)
        evaluations.update({"test": metrics_test})

    # Prediction
    results = predict_model(
        model,
        input_layers,
        target_shape=(meta["height"], meta["width"])
    )

    # Collect metadata
    cma_metadata = {
        "cma_id": cma_id,
        "model_run_id": model_run_id
    }

    network_arch = {
        "core_units": core_units,
        "head_units": head_units,
    }

    label_data = [
        ("initial", _get_label_info(y_prepared)),
        ("sampled_negatives", _get_label_info(y_sampled)),
        ("train_split", _get_label_info(y_train_split)),
        ("test_split", _get_label_info(y_test_split)),
        ("final", _get_label_info(y_train)),
    ]

    model_metadata = _collect_settings(
        cma_metadata=cma_metadata,
        train_config=train_config,
        network_arch=network_arch,
        label_data=label_data,
    )

    # Set output locations for files
    metrics_file_name = "metrics.json"
    meta_file_name = "settings.json"

    output_locations = (
        output_folder,
        raster_folder,
        os.path.join(output_folder, metrics_file_name),
        os.path.join(output_folder, meta_file_name),
    )

    # Save metrics and model metadata
    write_json(output_folder, metrics_file_name, evaluations)
    write_json(output_folder, meta_file_name, model_metadata)

    # Save rasters
    target_meta = meta.copy()
    target_meta.update(
        {
            "nodata": -99999
        }
    )

    for result, name in zip(results, ["likelihoods", "uncertainties"]):
        out_file = os.path.join(raster_folder, f"{name}.tif")
        out_array, out_meta = prepare_output(result, target_meta)
        save_raster(out_array, out_meta, out_file)

    # Collect results
    out_layers = _collect_results(
        cma_id=cma_id,
        model_run_id=model_run_id,
        output_locations=output_locations,
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


def _collect_settings(
    cma_metadata: Dict,
    train_config: Dict,
    network_arch: Dict,
    label_data: List[Tuple[str, Dict]],
):
    """
    Create a dictionary containing all settings.

    Args:
        cma_metadata: Metadata for the CMA.
        train_config: Configuration for training.
        network_arch: Network architecture.
        label_data: Data and descriptions for labels.

    Returns:
        dict: Dictionary containing all settings.
    """
    labels = {}
    for description, data in label_data:
        labels.update(
            {
                description: data
            }
        )

    metadata = {
        "cma": cma_metadata,
        "train_config": train_config,
        "network_architecture": network_arch,
        "labels": labels,
    }

    return metadata


def _collect_results(
    cma_id: str,
    model_run_id: str,
    output_locations: Tuple[str, ...],
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
    output_folder, raster_folder, metrics_file, settings_file = output_locations

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
    layers_list = []

    # Add raster results
    for file in results_file_list:
        file_stem = Path(file).stem.lower()
        meta = init_meta.copy()

        for key, value in results.items():
            if key == file_stem:
                meta.update(value)

                layers_list.append(
                    (file, meta)
                )

    # Add metrics
    update_meta = {
        "output_type": "metrics",
        "title": "Archive containing the calculated metrics"
    }

    layers_list = prepare_output_layers(
        layers=layers_list,
        files=metrics_file,
        meta=(init_meta, update_meta),
        output=(output_folder, "metrics.zip")
    )

    # Add metadata and settings
    update_meta = {
        "output_type": "metadata",
        "title": "Archive containing model metadata and settings"
    }

    layers_list = prepare_output_layers(
        layers=layers_list,
        files=settings_file,
        meta=(init_meta, update_meta),
        output=(output_folder, "settings.zip")
    )

    # Create CDR object
    prospectivity_output_layers = []
    for layer in layers_list:
        layer_path = layer[0]
        layer_meta = layer[1]

        layer_object = ProspectivityOutputLayer(**layer_meta)
        prospectivity_output_layers.append(
            (layer_path, layer_object)
        )

    return prospectivity_output_layers
