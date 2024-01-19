import multiprocessing as mp
import time
import warnings

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy
import rasterio
from rasterio import warp
from tqdm import tqdm

from beak.utilities.io import (
    create_file_folder_list,
    create_file_list,
    check_path,
    load_raster,
    save_raster,
)

from beak.utilities.misc import replace_invalid_characters

# References
# Some non-trivial functionalities were adapted from other sources.
# The original sources are listed below and referenced in the code as well.
#
# EIS toolkit:
# GitHub repository https://github.com/GispoCoding/eis_toolkit under EUPL-1.2 license.


def fill_nodata_with_mean(
    array: np.ndarray,
    nodata_value: np.number,
    size: int = 3,
    num_nan_max: int = 4,
) -> np.ndarray:
    """
    Fill nodata values with the mean from surrounding cells.

    Args:
        array (np.ndarray): Input array with nodata values.
        nodata_value (np.number): Value representing nodata in the array.
        size (int): Size of the kernel used for calculating the mean. Defaults to 3.
        num_nan_max (int): Maximum number of nodata cells allowed in the kernel neighborhood for mean calculation. Defaults to 4.

    Returns:
        np.ndarray: Array with nodata values imputed by the mean from surrounding cells.
    """
    # Set kernel size
    kernel = np.ones((size, size))

    # Create mask for nodata values and convert to int
    nan_mask = np.isin(array, nodata_value)

    # Create array for sum of np.nan values in kernel neighborhood
    nan_sum = scipy.ndimage.generic_filter(
        nan_mask.astype(int), np.sum, footprint=kernel, mode="constant", cval=0
    )

    # Create combined masked with certain amount of nodata cells allowed for mean calculation
    nan_sum_mask = np.logical_and(nan_mask, nan_sum <= num_nan_max)

    # Initialize output array
    out_array = np.where(nan_mask, np.nan, array)

    # Calculate mean for each cell in kernel neighborhood based on nan_sum_mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        out_array = np.where(
            nan_sum_mask,
            scipy.ndimage.generic_filter(
                out_array, np.nanmean, footprint=kernel, mode="reflect"
            ),
            out_array,
        )

    return np.where(np.isnan(out_array), nodata_value, out_array)


# region: reproject raster data
def _reproject_raster_process(
    file: Path,
    input_folder: Path,
    output_folder: Path,
    target_epsg: int,
    target_resolution: Optional[np.number],
    resampling_method: warp.Resampling,
):
    """Run reprojection process for a single raster file.

    Args:
        file (Path): The path to the input raster file.
        input_folder (Path): The path to the input folder.
        output_folder (Path): The path to the output folder.
        target_epsg (int): The target EPSG code for the reprojection.
        target_resolution (Optional[np.number]): The target resolution for the reprojection.
        resampling_method (warp.Resampling): The resampling method to use.

    Returns:
        None
    """
    raster = load_raster(file)
    out_file = output_folder / file.relative_to(Path(input_folder))
    check_path(out_file.parent)
    out_array, out_meta = _reproject_raster_core(
        raster, target_epsg, target_resolution, resampling_method
    )

    save_raster(
        out_file,
        out_array,
        target_epsg,
        out_meta["height"],
        out_meta["width"],
        raster.nodata,
        out_meta["transform"],
    )


def _reproject_raster_core(
    raster: rasterio.io.DatasetReader,
    target_crs: int,
    target_resolution: Optional[np.number],
    resampling_method: warp.Resampling,
) -> Tuple[np.ndarray, dict]:
    """
    Reproject a raster to a new coordinate reference system (CRS) and resolution.

    Adapted function from EIS Toolkit (main branch as of 2023-11-17).

    Args:
        raster (rasterio.io.DatasetReader): The input raster to be reprojected.
        target_crs (int): The EPSG code of the target CRS.
        target_resolution (Optional[np.number]): The target resolution of the reprojected raster.
        resampling_method (warp.Resampling): The resampling method to be used during reprojection.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing the reprojected image as a NumPy array and the metadata of the reprojected raster.
    """

    src_arr = raster.read()
    dst_crs = rasterio.crs.CRS.from_epsg(target_crs)

    dst_transform, dst_width, dst_height = warp.calculate_default_transform(
        raster.crs,
        dst_crs,
        raster.width,
        raster.height,
        resolution=target_resolution,
        *raster.bounds,
    )

    # Initialize output raster
    dst = np.empty((raster.count, dst_height, dst_width))
    dst.fill(raster.meta["nodata"])

    out_image = warp.reproject(
        source=src_arr,
        src_transform=raster.transform,
        src_crs=raster.crs,
        destination=dst,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=raster.meta["nodata"],
        dst_nodata=raster.meta["nodata"],
        resampling=resampling_method,
    )[0]

    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
        }
    )

    return out_image.astype(src_arr.dtype), out_meta


def reproject_raster(
    input_folder: Path,
    output_folder: Path,
    target_epsg: int,
    target_resolution: Optional[np.number] = None,
    resampling_method: warp.Resampling = warp.Resampling.nearest,
    n_workers: int = mp.cpu_count(),
):
    """
    Reprojects rasters from the input folder to the output folder using the specified target EPSG code.

    Args:
        input_folder (Path): The path to the input folder containing the rasters.
        output_folder (Path): The path to the output folder where the reprojected rasters will be saved.
        target_epsg (int): The EPSG code of the target coordinate reference system (CRS).
        target_resolution (Optional[np.number]): The target resolution of the reprojected rasters. Defaults to None.
        resampling_method (warp.Resampling): The resampling method to use during reprojection. Defaults to warp.Resampling.nearest.
        n_workers (int): The number of worker processes to use for parallel processing. Defaults to the number of CPU cores.
    """
    # Show selected folder
    print(f"Selected folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Get all folders in the root folder
    folders, _ = create_file_folder_list(Path(input_folder))
    print(f"Total of folders found: {len(folders)}")

    # Load rasters for each folder
    file_list = []

    with mp.Pool(n_workers) as pool:
        results = pool.map(create_file_list, folders)

    for result in results:
        file_list.extend(result)

    # Show results
    print(f"Files loaded: {len(file_list)}")

    # Set args list
    args_list = [
        (
            file,
            input_folder,
            output_folder,
            target_epsg,
            target_resolution,
            resampling_method,
        )
        for file in file_list
    ]

    # Run reprojection
    with mp.Pool(n_workers) as pool:
        with tqdm(total=len(args_list), desc="Processing files") as pbar:
            for _ in pool.starmap(_reproject_raster_process, args_list):
                pbar.update(1)
                time.sleep(0.1)


# endregion
