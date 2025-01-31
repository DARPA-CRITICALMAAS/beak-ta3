import os
import sys
import json
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any, Dict

from beak.utilities.file_io import write_json, write_csv, write_gpkg
from beak.utilities.vector_processing import extract_values_from_points

from cdr_schemas.prospectivity_input import ProspectivityOutputLayer


def _get_data_folder(package_name = "beak", folder_name: str = "data") -> Any:
    """
    Obtain the path to the specified data folder.

    Args:
        folder_name: The name of the data folder. Default is "beak.data".

    Returns:
        The relative path to the specified data folder.
    """
    if sys.version_info < (3, 9):
        from importlib_resources import files
    else:
        from importlib.resources import files

    return files(package_name) / folder_name


def _extract_rows(
    data: pd.DataFrame,
    attribute: str,
) -> List:
    """
    Extract specific information from the provided data frame.

    Args:
        data: The DataFrame containing the data.
        attribute: The column name to extract information from for each row.

    Returns:
        A list containing the extracted information.
    """
    extractions = []

    for index, row in data.iterrows():
        information = row[attribute]
        extractions.append(information)

    return extractions


def create_file_list(
    folder: Union[str, Path],
    file_suffix: Union[str, Tuple, None] = (".tif", ".tiff"),
    file_prefix: Union[str, Tuple, None] = None,
) -> List[str]:
    """
    Create a list of files in the specified folder with the given extensions.

    Args:
        folder: The folder path to search for files.
        file_suffix: Sequence of file extensions to include. Defaults to [".tif", ".tiff"].
        file_prefix: Prefixes of files to filter. Defaults to None.

    Returns:
        A list of file paths found.
    """
    folder = Path(folder)
    file_list = [str(file) for file in folder.glob("*")]

    return _filter_files(
        file_list,
        file_suffix=file_suffix,
        file_prefix=file_prefix,
    )


def _filter_files(
    file_list: List[str],
    file_suffix: Union[str, Tuple, None],
    file_prefix: Union[str, Tuple, None],
) -> List[str]:
    """
    Filter files from the provided list of files.

    Args:
        file_list: List of file paths.
        file_suffix: Suffixes of files to filter.
        file_prefix: Prefixes of files to filter.

    Raises:
        AssertionError: If no files are found.

    Returns:
        List of file paths that match the specified criteria.
    """
    suffix_match, prefix_match = True, True
    filtered_files = []

    for file in file_list:
        file_name = str(
            Path(file).name
        ).lower()

        if file_suffix is not None:
            file_suffix = tuple(ext.lower() for ext in file_suffix) if isinstance(file_suffix, tuple) else file_suffix.lower()
            suffix_match = file_name.endswith(file_suffix)

        if file_prefix is not None:
            file_prefix = tuple(prefix.lower() for prefix in file_prefix) if isinstance(file_prefix, tuple) else file_prefix.lower()
            prefix_match = file_name.startswith(file_prefix)

        if suffix_match and prefix_match:
            filtered_files.append(file)

    assert filtered_files, "No files found."
    return filtered_files


def _download_cdr_files(
    download_urls: List[str],
    download_folder: Path,
    verify_ssl: bool,
) -> List[str]:
    """
    Download files from the provided list of URLs to a specified folder.

    Args:
        download_urls: List of URLs from which to download files.
        download_folder: Path to the folder where the files will be downloaded.
        verify_ssl: Whether to verify SSL certificates during the download.

    Returns:
        List of file paths that were downloaded.
    """
    download_folder.mkdir(parents=True, exist_ok=True)

    file_list = []
    for url in download_urls:
        url_name = Path(url).name
        download_file_path = download_folder / url_name

        if not os.path.exists(download_file_path):
            response = requests.get(url, verify=verify_ssl)
            with open(download_file_path, "wb") as file:
                    file.write(response.content)

        file_list.append(
            str(download_file_path)
        )

    return file_list


def _create_zip_from_files(
    file_list: List[Union[str, Path]],
    archive_path: Union[str, Path]
):
    """
    Create a zip file from a list of files.

    Args:
        file_list: The list of input files to be saved.
        archive_path: The path to the output zip file.

    Returns:
        None
    """
    file_list = [Path(file) for file in file_list if os.path.exists(file)]
    archive_path = Path(archive_path)

    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in file_list:
            archive.write(filename=file, arcname=file.name)

    if not file_list:
        print(f"Warning: Empty archive {archive_path.name} created.")


def delete_files(file_list: List[str]) -> None:
    """
    Delete files from the specified list.

    Args:
        file_list: List of file paths to delete.

    Returns:
        None
    """
    for file in file_list:
        try:
            os.remove(file)
        except:
            print(f"Failed to delete file: {file}")


def _filter_layers_from_payload(
    payload: pd.DataFrame,
    label_column: Optional[str],
    filter_labels: bool = False
) -> pd.DataFrame:
    """
    Filter evidence layers from configuration file.

    Args:
        payload: DataFrame containing the payload.
        label_column: Column name containing labels.
        filter_labels: Whether to filter layers based on labels. Defaults to False.

    Returns:
        Filtered dataframe containing evidence layers or label file.
    """
    layers = payload["event"]["evidence_layers"]
    layers = pd.json_normalize(layers)

    if label_column in layers.columns:
        layers = layers[
            layers[label_column] == filter_labels
        ]

    return layers


def prepare_output_layers(
    layers: List[Tuple[str, Dict]],
    files: Union[str, List[str]],
    meta: Tuple[Dict, Dict],
    output: Optional[Tuple[str, str]],
) -> List[Tuple[str, Dict]]:
    """
    Concatenate file paths and metadata for output layers.

    If no output folder is provided, no archive will be created.
    Single files must be provided as a string.

    Args:
        layers: List of tuples containing layer names and metadata.
        files: Path to the file or list of files to be included in the archive.
        meta: Initial and updated metadata for the output layers.
        output: Output folder and archive name.

    Returns:
        List of tuples containing layer names and updated metadata.
    """
    init_meta, update_meta = meta
    meta = init_meta.copy()
    meta.update(update_meta)

    if output is not None:
        output_folder, archive_name = output
        file_list = [files] if not isinstance(files, list) else files

        file_path = str(
            os.path.join(output_folder, archive_name)
        )

        _create_zip_from_files(
            file_list=file_list,
            archive_path=file_path
        )
    else:
        file_path = files

    layers.append((file_path, meta)) if os.path.exists(file_path) else layers

    return layers


def _create_model_run_config(
    model_run: Tuple[str, str],
    input_data: Tuple[List[str], Optional[str]],
    train_config: Dict,
    **kwargs
) -> Dict:
    """
    Create a configuration file for the given CMA and model run ID.

    Args:
        model_run: Tuple containing CMA ID and model run ID.
        input_data: Tuple containing list of input layers and optional label file path.
        train_config: Dictionary containing training configuration.
        **kwargs: Additional key-value pairs to be included in the export.
    Returns:
        Dictionary containing the the model run configuration.
    """
    cma_id, model_run_id = model_run
    layers, labels = input_data
    
    model_config = {
        "cma": {
            "cma_id": cma_id,
            "model_run_id": model_run_id
        },
        "train_config": train_config,
    }
    model_config.update(kwargs)

    layers_dict = {}
    for i, file in enumerate(layers):
        layers_dict.update({
            f"layer_{i:03d}": os.path.basename(file)
        })

    model_config.update({
        "input_data": {
            "labels": os.path.basename(labels) if labels else None,
            "layers": layers_dict
        }
    })
    
    return model_config


def update_config_json(
    file_path: Union[Path, str],
    config_loc: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Load the config JSON file, update parameters, and save the changes.

    Args:
        file_path: Path to the config file.
        config_loc: List of keys to navigate through the JSON structure. Defaults to None.
        **kwargs: Keyword arguments for parameters to update.

    Returns:
        None
    """
    if config_loc is None:
        config_loc = ["events", "payload", "train_config"]

    with open(file_path, "r") as file:
        config = json.load(file)

    config_key = config
    for key in config_loc:
        config_key = config_key[key]

    config_key.update({key: value for key, value in kwargs.items()})
    file.close()

    write_json(
        os.path.dirname(file_path),
        os.path.basename(file_path),
        config
    )


def _add_raster_results_to_output_layers(
    file_list: List[str],
    results: Dict,
    init_meta: Dict,
) -> List[Tuple[str, Dict]]:
    """
    Add raster results to output layers.

    Args:
        file_list: List of file paths.
        results: Dictionary containing raster results.
        init_meta: Initial metadata for the output layers.

    Returns:
        List of tuples containing file paths and updated metadata for output layers.
    """
    layers_list = []

    for file in file_list:
        file_stem = Path(file).stem.lower()
        meta = init_meta.copy()

        for key, value in results.items():
            if key == file_stem:
                meta.update(value)

                layers_list.append(
                    (file, meta)
                )

    return layers_list


def export_train_test_splits(
    src_data: Optional[Tuple[List[np.ndarray], List[str]]],
    split_data: Tuple[List[pd.DataFrame], List[str]],
    template: Tuple[np.ndarray, Dict],
    output_folder: str,
    geometry: str = "geometry",
    drop_columns: Optional[List[str]] = None
) -> None:
    """
    Export train/test splits as CSV and GeoPackage files.

    Args:
        src_data: Optional tuple containing source arrays and names.
        split_data: Tuple containing split dataframes and names.
        template: Template data and metadata.
        output_folder: Output folder path.
        geometry: Name of the geometry column. Defaults to "geometry".
        drop_columns: List of columns to drop from the output CSV and GeoPackage files. Defaults to None.

    Returns:
        None
    """
    src_arrays, src_names = src_data
    split_sets, split_names = split_data
    drop_columns = drop_columns or []

    for split, file_name in zip(split_sets, split_names):
        out_data = (
            extract_values_from_points(split, src_arrays, src_names, template)
            if src_arrays and src_names else split
        )

        output_path = os.path.join(output_folder, file_name)

        drop_cols_csv = drop_columns + [geometry]
        drop_cols_gpkg = [col for col in drop_columns if col != geometry]

        write_csv(src_data=out_data, file_path=f"{output_path}.csv", drop_columns=drop_cols_csv)
        write_gpkg(src_data=out_data, file_path=f"{output_path}.gpkg", drop_columns=drop_cols_gpkg)


def _create_prospectivity_output_layers(
    layers_list: List[Tuple[str, Dict]],
) -> List[Tuple[str, ProspectivityOutputLayer]]:
    """
    Create ProspectivityOutputLayer objects from the provided layers and metadata.

    Args:
        layers_list: List of tuples containing file paths and metadata for output layers.

    Returns:
        List of tuples containing file paths and ProspectivityOutputLayer objects.
    """
    prospectivity_output_layers = []
    for layer in layers_list:
        layer_path = layer[0]
        layer_meta = layer[1]

        layer_object = ProspectivityOutputLayer(**layer_meta)
        prospectivity_output_layers.append(
            (layer_path, layer_object)
        )

    return prospectivity_output_layers


def _create_runtime_stats(
    input_size: int,
    meta: Dict,
    runtime: float,
    **kwargs,
) -> Dict:
    """
    Create a dictionary containing runtime statistics for a model run.

    Args:
        input_size: Number of input layers used in the model.
        meta: Dictionary containing metadata about the model run, including width and height.
        runtime: Total runtime of the model in seconds.
        **kwargs: Additional keyword arguments to include in the model run statistics.

    Returns:
        A dictionary containing model run information and runtime statistics.
    """
    width, height = meta["width"], meta["height"]

    runtime_stats = {
        "model_run": {
            "extent_width": width,
            "extent_height": height,
            "extent_pixels": width * height,
            "input_layers": input_size,
        }
    }
    runtime_stats["model_run"].update(kwargs)

    runtime_stats.update({
        "runtime_stats": {
            "runtime_seconds": round(runtime, 0),
            "runtime_minutes": round(runtime / 60, 1),
            "runtime_hours": round(runtime / 3600, 2)
        }
    })

    return runtime_stats
