import multiprocessing as mp
import time
import os
import warnings

from pathlib import Path
from typing import Optional, Tuple, Union, Sequence, List

import numpy as np
import geopandas as gpd
import scipy
import rasterio
from numbers import Number
from rasterio import warp
from rasterio.mask import mask
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

from beak.utilities.io import (
    create_file_folder_list,
    create_file_list,
    check_path,
    load_raster,
    save_raster,
)

from beak.utilities.io import load_raster, save_raster, copy_folder_structure


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
    """
    out_file = output_folder / file.relative_to(Path(input_folder))

    if not os.path.exists(out_file):
        raster = load_raster(file)
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
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            print("Starting parallel processing...")
            pool.starmap(_reproject_raster_process, args_list)
    else:
        print("Starting single processing...")
        for args in tqdm(args_list, desc="Reprojecting rasters"):
            _reproject_raster_process(*args)

    print("Done!")


# endregion


def _clip_raster_with_coords(
    input_raster: Union[str, Path],
    output_raster: Union[str, Path],
    bounds: Tuple[
        Optional[Number], Optional[Number], Optional[Number], Optional[Number]
    ],
    write_result: bool = True,
    return_result: bool = False,
    intermediate_result: Optional[Tuple[np.ndarray, dict]] = None,
) -> tuple[np.ndarray, dict]:
    """Clips the input raster using the provided coordinates."""
    if (
        write_result is True and not os.path.exists(output_raster)
    ) or return_result is True:
        left = bounds[0] if bounds[0] is not None else input_raster.bounds.left
        bottom = bounds[1] if bounds[1] is not None else input_raster.bounds.bottom
        right = bounds[2] if bounds[2] is not None else input_raster.bounds.right
        top = bounds[3] if bounds[3] is not None else input_raster.bounds.top

        window = input_raster.window(left, bottom, right, top)
        row_start, col_start, row_stop, col_stop = map(
            int,
            (
                window.row_off,
                window.col_off,
                window.row_off + window.height,
                window.col_off + window.width,
            ),
        )

        if intermediate_result is None:
            clipped_data = input_raster.read()
            clipped_meta = input_raster.meta.copy()
            clipped_meta.update(
                {
                    "transform": rasterio.windows.transform(
                        window, input_raster.transform
                    ),
                }
            )
        else:
            clipped_data = intermediate_result[0]
            clipped_meta = intermediate_result[1].copy()

        clipped_data = clipped_data[:, row_start:row_stop, col_start:col_stop]
        clipped_meta.update(
            {
                "driver": "GTiff",
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
            }
        )

        if write_result is True:
            check_path(output_raster.parent)
            with rasterio.open(output_raster, "w", **clipped_meta) as dst:
                dst.write(clipped_data)

        if return_result is True:
            return clipped_data, clipped_meta


def _clip_raster_with_shapefile(
    input_raster: rasterio.io.DatasetReader,
    output_raster: Union[str, Path],
    shapefile: Union[str, Path],
    query: Optional[str],
    all_touched: bool,
    write_result: bool = True,
    return_result: bool = False,
) -> tuple[np.ndarray, dict]:
    """Clips a raster with a shapefile using the specified query."""
    if (
        write_result is True and not os.path.exists(output_raster)
    ) or return_result is True:
        gdf = gpd.read_file(shapefile)
        gdf = gdf.query(query) if query is not None else gdf

        clipped_data, clipped_transform = mask(
            input_raster, gdf.geometry, crop=True, all_touched=all_touched
        )

        clipped_meta = input_raster.meta.copy()
        clipped_meta.update(
            {
                "driver": "GTiff",
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform,
            }
        )

        if write_result is True:
            check_path(output_raster.parent)
            with rasterio.open(output_raster, "w", **clipped_meta) as dst:
                dst.write(clipped_data)

        if return_result is True:
            return clipped_data, clipped_meta


def _clip_raster_process(
    file: Path,
    input_folder,
    output_folder: Union[str, Path],
    shapefile: Optional[Union[str, Path]],
    query: Optional[str],
    bounds: Optional[
        Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]
    ] = None,
    all_touched: bool = True,
):
    """
    Clips a raster file based on either a shapefile or bounding coordinates.

    Args:
        file (Path): The path to the input raster file.
        output_folder (Union[str, Path]): The folder where the clipped raster will be saved.
        shapefile (Optional[Union[str, Path]]): The path to the shapefile used for clipping. If None, bounds must be provided.
        query (Optional[str]): An optional query string to filter the shapefile features.
        bounds (Optional[Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]]): The bounding coordinates used for clipping. If None, shapefile must be provided.
        all_touched (bool): Whether to include all pixels touched by the shapefile features. Defaults to True.

    Raises:
        ValueError: If neither shapefile nor bounds are provided.

    """
    relative_file = file.relative_to(input_folder)
    output_raster = output_folder / relative_file

    raster = rasterio.open(file)
    if shapefile is not None and bounds is None:
        _clip_raster_with_shapefile(
            raster,
            output_raster,
            shapefile,
            query,
            all_touched,
            write_result=True,
            return_result=False,
        )
    elif shapefile is None and bounds is not None:
        _clip_raster_with_coords(
            raster,
            output_raster,
            bounds,
            write_result=True,
            return_result=False,
        )
    elif shapefile is not None and bounds is not None:
        clipped_raster, clipped_meta = _clip_raster_with_shapefile(
            raster,
            output_raster,
            shapefile,
            query,
            all_touched,
            write_result=False,
            return_result=True,
        )
        _clip_raster_with_coords(
            raster,
            output_raster,
            bounds,
            write_result=True,
            return_result=False,
            intermediate_result=(clipped_raster, clipped_meta),
        )
    else:
        raise ValueError("Either shapefile or bounds must be provided for clipping.")


def clip_raster(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    shapefile: Optional[Union[str, Path]],
    query: Optional[str],
    bounds: Optional[
        Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]
    ] = None,
    raster_extensions: List[str] = [".tif", ".tiff"],
    include_source: bool = True,
    all_touched: bool = True,
    n_workers: int = mp.cpu_count(),
):
    """
    Clips rasters within the specified input folder to the extent of a shapefile or a bounding box.

    Args:
        input_folder (Union[str, Path]): Path to the input folder containing the rasters.
        output_folder (Union[str, Path]): Path to the output folder where the clipped rasters will be saved.
        shapefile (Optional[Union[str, Path]]): Path to the shapefile used for clipping.
            If None, the rasters will be clipped to the specified bounding box.
        query (Optional[str]): Query string to filter the features in the shapefile.
            Only features that satisfy the query will be used for clipping. Ignored if shapefile is None.
        bounds (Optional[Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]]):
            Bounding box coordinates (minx, miny, maxx, maxy) used for clipping. Ignored if shapefile is not None.
        raster_extensions (List[str]): List of file extensions to consider as rasters. Default is [".tif", ".tiff"].
        include_source (bool): Flag indicating whether to include the input folder itself as a source for clipping. Default is True.
        all_touched (bool): Flag indicating whether to include all pixels touched by the shapefile or bounding box. Default is True.
        n_workers (int): Number of parallel workers to use for clipping. Default is the number of available CPU cores.
    """

    folders, _ = create_file_folder_list(Path(input_folder))
    if include_source is True:
        folders.insert(0, input_folder)

    file_list = []
    for folder in folders:
        folder_file_list = create_file_list(folder, raster_extensions)
        file_list.extend(folder_file_list)

    args_list = [
        (
            file,
            input_folder,
            output_folder,
            shapefile,
            query,
            bounds,
            all_touched,
        )
        for file in file_list
    ]

    # Run clipping operation
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            print("Starting parallel processing...")
            pool.starmap(_clip_raster_process, args_list)
    else:
        print("Starting single processing...")
        for args in tqdm(args_list, desc="Clipping rasters"):
            _clip_raster_process(*args)

    print("Done!")


# region: Test code

# endregion
