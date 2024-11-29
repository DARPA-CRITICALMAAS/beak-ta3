import os
import re
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from beartype.typing import Optional, Union, Sequence, Tuple
from beak.experimental.io import create_file_list


def replace_invalid_characters(string: str, prefix: Optional[str] = None) -> str:
    """
    Replace invalid characters in a string with an underscore.
    Multiple consecutive underscores will be cropped to one.
    If a prefix is provided, it will be added to the beginning of the string.

    Args:
        string (str): Input string.
        prefix (Optional[str]): Prefix to add to the beginning of the string.

    Returns:
        str: String with replaced characters.
    """
    # Replace invalid characters
    string = re.sub(r"[ /().:,<>]", "_", string)

    # Remove leading, trailing and ending underscores
    string = re.sub(r"(_+)", "_", string)
    string = re.sub(r"^_|_$", "", string)

    # Add prefix if provided
    if prefix is not None:
        string = prefix + "_" + string

    return string


def create_tree(
    path: Optional[str] = None,
    parent: bool = False,
    depth: int = 0,
    parent_prefix: str = "",
    is_last: bool = False,
    excluded_folders: list = ["eis_toolkit", "__pycache__"],
):
    """Create a tree structure of a given directory."""
    path = Path.cwd() if path is None else Path(path)
    path = path.parent if parent is True and depth == 0 else path

    folders = [
        entry
        for entry in os.listdir(path)
        if os.path.isdir(os.path.join(path, entry))
        and entry not in excluded_folders
        and not entry.startswith((".", "_"))
    ]

    # Sort folders for consistency
    folders.sort()

    tree = ""
    entry_prefix = "└── " if is_last else "├── "

    # Display the current directory
    tree += parent_prefix + entry_prefix + os.path.basename(path) + "/\n"

    # Prepare the prefix for subdirectories
    prefix = parent_prefix + ("    " if is_last else "│   ")

    if depth < 2:
        for i, folder in enumerate(folders):
            full_path = os.path.join(path, folder)

            # Check if it is the last entry in the current directory
            last_entry = i == len(folders) - 1

            tree += create_tree(
                full_path, parent, depth + 1, prefix, last_entry, excluded_folders
            )

    return tree


def create_raster_report(
    path: Union[Path, str] = None,
    file_extensions: Sequence[str] = [".tif", ".tiff"],
    recursive_search: bool = True,
    out_file: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    Create a report of all raster files in a given directory.

    Parameters:
    path (Union[Path, str], optional): The path to the directory containing raster files. Defaults to None, which means the current working directory.
    file_extensions (list, optional): A list of file extensions to consider as raster files. Defaults to [".tif", ".tiff"].
    recursive_search (bool, optional): Whether to search for raster files recursively in subdirectories. Defaults to True.
    out_file (Optional[Union[Path, str]], optional): The path to save the generated report as a CSV file. Defaults to None, which means the report will not be saved.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the report of raster files.
    """
    path = Path(path)
    file_list = create_file_list(
        path, extensions=file_extensions, recursive=recursive_search
    )

    report = pd.DataFrame()
    for file in tqdm(file_list):
        raster = rasterio.open(file)

        for band in range(0, raster.count):
            array = raster.read(band + 1)
            array = np.where(array == raster.nodatavals[band], np.nan, array)

            decimals = 6
            band_report = {
                "file_name": file.name,
                "crs": raster.crs,
                "width": round(raster.width, decimals),
                "height": round(raster.height, decimals),
                "count": raster.count,
                "band": band + 1,
                "single_band": True if raster.count == 1 else False,
                "nodata": raster.nodatavals[band],
                "dtype": raster.dtypes[band],
                "cellsize_x": round(abs(raster.transform.a), decimals),
                "cellsize_y": round(abs(raster.transform.e), decimals),
                "value_min": round(np.nanmin(array), decimals),
                "value_max": round(np.nanmax(array), decimals),
                "value_mean": round(np.nanmean(array), decimals),
                "value_median": round(np.nanmedian(array), decimals),
                "value_std": round(np.nanstd(array), decimals),
                "file_path": file,
                "transform": " ".join(map(str, raster.transform)),
            }

            band_report = pd.DataFrame.from_dict(
                band_report, orient="index"
            ).transpose()

            report = pd.concat([report, band_report], ignore_index=True)

    if out_file is not None:
        report.to_csv(out_file, index=False)

    return report


def update_model_config_core(model: dict, changes: Tuple[str, Optional[str]]) -> dict:
    """
    Update the core model configuration using evidence layers.

    Args:
        model (dict): The input dictionary.
        changes (Tuple[str, str]): A tuple containing the old evidence layer

    Returns:
        out_dict (dict): The updated dictionary with evidence layers replaced.
    """
    out_dict = {}
    old_evidence, new_evidence = changes

    for key, value in model.items():
        if key == old_evidence:
            out_dict[new_evidence] = value
        else:
            out_dict[key] = value

    return out_dict


def update_model_config(model: dict, changes: Sequence[tuple[str, str]]) -> dict:
    """
    Replace evidence layers in a dictionary.

    Args:
        dict (dict): The input dictionary.
        changes (Sequence[Tuple[str, str]]): A Sequence of tuples containing the old evidence layer
            and the new evidence combinations.

    Returns:
        dict: The updated dictionary with evidence layers replaced.
    """
    out_dict = model.copy()

    for change in changes:
        out_dict = update_model_config_core(out_dict, change)

    return out_dict
