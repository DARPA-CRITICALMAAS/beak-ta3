import multiprocessing as mp
import time
from pathlib import Path
from typing import Tuple, Literal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import rasterio
from tqdm import tqdm

from utilities.io import (
    create_file_folder_list,
    create_file_list,
    check_path,
    load_raster,
    save_raster,
)

# References
# Some non-trivial functionalities were adapted from other sources.
# The original sources are listed below and referenced in the code as well.
#
# EIS toolkit:
# GitHub repository https://github.com/GispoCoding/eis_toolkit under EUPL-1.2 license.


# region: reproject raster data
def _scale_raster_process(
    file: Path,
    input_folder: Path,
    output_folder: Path,
    method: str,
):
    """Run scaling process for a single raster file.

    Args:
        file (Path): The path to the input raster file.
        input_folder (Path): The path to the input folder.
        output_folder (Path): The path to the output folder.
        method (str): The scaling method to be used.

    Returns:
        None
    """
    raster = load_raster(file)
    out_file = output_folder / file.relative_to(Path(input_folder))
    check_path(out_file.parent)
    out_array = _scale_raster_core(raster, method)

    save_raster(
        out_file,
        out_array,
        raster.crs.to_epsg(),
        raster.height,
        raster.width,
        raster.nodata,
        raster.transform,
    )


def _scale_raster_core(
    raster: rasterio.io.DatasetReader,
    method: str,
) -> Tuple[np.ndarray, dict]:
    """
    Scale a raster to a new range.

    Args:
        raster (rasterio.io.DatasetReader): The input raster to be scaled.
        method (Literal[str]): The scaling method to be used. Options are "minmax" for min-max scaling and "standard" for standard scaling.

    Returns:
        np.ndarray: Numpy array of the rescaled raster.
    """
    src_array = raster.read()
    src_array = src_array.squeeze()
    src_array = np.where(src_array == raster.nodata, np.nan, src_array)

    if method == "minmax":
        scaler = MinMaxScaler()
        out_array = scaler.fit_transform(src_array.reshape(-1, 1))
    elif method == "standard":
        scaler = StandardScaler()
        out_array = scaler.fit_transform(src_array.reshape(-1, 1))

    out_array = np.where(np.isnan(out_array), raster.nodata, out_array)
    out_array = np.reshape(out_array, src_array.shape)
    return out_array.astype(src_array.dtype)


def scale_raster(
    input_folder: Path,
    output_folder: Path,
    method: Literal["minmax", "standard"],
    n_workers: int = mp.cpu_count(),
):
    """
    Reprojects rasters from the input folder to the output folder using the specified target EPSG code.

    Args:
        input_folder (Path): The path to the input folder containing the rasters.
        output_folder (Path): The path to the output folder where the reprojected rasters will be saved.
        method (Literal[str]): The scaling method to be used. Options are "minmax" for min-max scaling and "standard" for z-score scaling.
        n_workers (int): The number of worker processes to use for parallel processing. Defaults to the number of CPU cores.
    """
    # Show selected folder
    print(f"Selected folder: {input_folder.resolve()}")

    # Get all folders in the root folder
    folders, files = create_file_folder_list(Path(input_folder))
    print(f"Total of folders found: {len(folders)}")

    # Show results
    print(f"Files loaded: {len(files)}")

    # Set args list
    args_list = [
        (
            file,
            input_folder,
            output_folder,
            method,
        )
        for file in files
    ]

    # Check output folder
    check_path(output_folder)

    # Run reprojection
    with mp.Pool(n_workers) as pool:
        with tqdm(total=len(args_list), desc="Processing files") as pbar:
            for _ in pool.starmap(_scale_raster_process, args_list):
                pbar.update(1)
                time.sleep(0.1)


# endregion
