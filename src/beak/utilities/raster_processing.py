import multiprocessing as mp
import os
import warnings

from pathlib import Path
from beartype.typing import Optional, Tuple, Union, Sequence, List, Literal

import numpy as np
import geopandas as gpd
import rasterio.coords
import scipy
import rasterio
import math

from numbers import Number
from rasterio import warp
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.crs import CRS
from tqdm import tqdm

from beak.utilities.io import (
    create_file_folder_list,
    create_file_list,
    check_path,
    load_raster,
    save_raster,
)

from beak.utilities.io import load_raster, save_raster
from beak.utilities.checks import check_grid_alignment

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
    target_crs: Union[int, rasterio.crs.CRS],
    target_resolution: Optional[np.number],
    resampling_method: warp.Resampling,
    resampling_mode: str,
    snap_to_origin: Union[str, Path, Tuple[Number, Number]],
):
    """Run reprojection process for a single raster file.

    Args:
        file (Path): The path to the input raster file.
        input_folder (Path): The path to the input folder.
        output_folder (Path): The path to the output folder.
        target_crs (Union[int, rasterio.crs.CRS]): Target coordinate reference system (CRS).
        target_resolution (Optional[np.number]): The target resolution for the reprojection.
        resampling_method (warp.Resampling): The resampling method to use.
        snap_to_origin (Union[str, Path, Tuple[Number, Number]]): The origin to snap to.
    """
    out_file = output_folder / file.relative_to(Path(input_folder))

    if not os.path.exists(out_file):
        raster = load_raster(file)
        snap_to_origin = load_raster(snap_to_origin) if isinstance(snap_to_origin, (str, Path)) else snap_to_origin
        check_path(Path(os.path.dirname(out_file)))
        out_array, out_meta = _reproject_raster_core(
            raster,
            target_crs,
            target_resolution,
            resampling_method,
            resampling_mode,
            snap_to_origin,
        )

        save_raster(
            out_file,
            out_array,
            out_meta["crs"],
            out_meta["height"],
            out_meta["width"],
            raster.nodata,
            out_meta["transform"],
        )


def _reproject_raster_core(
    raster: rasterio.io.DatasetReader,
    target_crs: Union[int, rasterio.crs.CRS],
    target_resolution: Optional[np.number],
    resampling_method: warp.Resampling,
    resampling_mode: str,
    snap_to_origin: Union[rasterio.io.DatasetReader, Tuple[Number, Number]],
) -> Tuple[np.ndarray, dict]:
    """
    Reproject a raster to a new coordinate reference system (CRS) and resolution.

    Adapted function from EIS Toolkit (main branch as of 2023-11-17).

    Args:
        raster (rasterio.io.DatasetReader): The input raster to be reprojected.
        target_crs (Union[int, rasterio.crs.CRS]): Target coordinate reference system (CRS).
        target_resolution (Optional[np.number]): The target resolution of the reprojected raster.
        resampling_method (warp.Resampling): The resampling method to be used during reprojection.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing the reprojected image as a NumPy array and the metadata of the reprojected raster.
    """

    src_arr = raster.read()

    if isinstance(target_crs, int):
        dst_crs = CRS.from_epsg(target_crs)
    else:
        dst_crs = target_crs

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

    if resampling_mode == "auto":
        if np.issubdtype(src_arr.dtype, np.integer):
            resampling_method = warp.Resampling.nearest
        elif np.issubdtype(src_arr.dtype, np.floating):
            resampling_method = warp.Resampling.bilinear

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

    if snap_to_origin is not None:
        out_image, out_meta = snap_raster(
            raster=(out_image, out_meta), snap_to=snap_to_origin
        )

    return out_image.astype(src_arr.dtype), out_meta


def reproject_raster(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    target_crs: Union[int, rasterio.crs.CRS],
    target_resolution: Optional[np.number] = None,
    resampling_method: warp.Resampling = warp.Resampling.nearest,
    resampling_mode: Literal["manual", "auto"] = "manual",
    include_source: bool = False,
    n_workers: int = mp.cpu_count(),
    snap_to_origin: Optional[Union[str, Path, Tuple[Number, Number]]] = None,
):
    """
    Reprojects rasters from the input folder to the output folder using the specified target EPSG code.

    Args:
        input_folder (Path): The path to the input folder containing the rasters.
        output_folder (Path): The path to the output folder where the reprojected rasters will be saved.
        target_crs (Union[int, rasterio.crs.CRS]): The target coordinate reference system (CRS).
        target_resolution (Optional[np.number]): The target resolution of the reprojected rasters. Defaults to None.
        resampling_method (warp.Resampling): The resampling method to use during reprojection. Defaults to warp.Resampling.nearest.
        resampling_mode (Literal["manual", "auto"]): Uses "nearest" for integers and "bilinear" for floats.
            Overwrites resampling_method if set to "auto". Defaults to "manual".
        n_workers (int): The number of worker processes to use for parallel processing. Defaults to the number of CPU cores.
        snap_to_origin (Optional[Union[str, Tuple[Number, Number]]]): Path to raster or origin to snap to. Defaults to None.
    """
    # Show selected folder
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    print(f"Selected folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Get all folders in the root folder
    folders, _ = create_file_folder_list(Path(input_folder))
    print(f"Total of folders found: {len(folders)}")

    if include_source is True:
        folders.insert(0, input_folder)

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
            target_crs,
            target_resolution,
            resampling_method,
            resampling_mode,
            snap_to_origin,
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
    input_raster: rasterio.io.DatasetReader,
    bounds: Tuple[
        Optional[Number], Optional[Number], Optional[Number], Optional[Number]
    ],
    intermediate_result: Optional[Tuple[np.ndarray, dict]] = None,
) -> tuple[np.ndarray, dict]:
    """Clips the input raster using the provided coordinates."""
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
                "transform": rasterio.windows.transform(window, input_raster.transform),
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
    return clipped_data, clipped_meta


def _clip_raster_with_shapefile(
    input_raster: rasterio.io.DatasetReader,
    shapefile: Union[str, Path],
    query: Optional[str],
    all_touched: bool,
) -> tuple[np.ndarray, dict]:
    """Clips a raster with a shapefile using the specified query."""
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
    return clipped_data, clipped_meta


def _clip_raster_process(
    file: Path,
    input_folder: Optional[Path],
    output_folder: Optional[Union[str, Path]],
    shapefile: Optional[Union[str, Path]],
    query: Optional[str],
    bounds: Optional[
        Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]
    ],
    all_touched: bool,
):
    """
    Clips a raster file based on either a shapefile or bounding coordinates.

    Args:
        file (Path): The path to the input raster file.
        output_folder (Optional[Union[str, Path]]): The folder where the clipped raster will be saved.
        shapefile (Optional[Union[str, Path]]): The path to the shapefile used for clipping. If None, bounds must be provided.
        query (Optional[str]): An optional query string to filter the shapefile features.
        bounds (Optional[Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]]): The bounding coordinates used for clipping. If None, shapefile must be provided.
        all_touched (bool): Whether to include all pixels touched by the shapefile features. Defaults to True.

    Raises:
        ValueError: If neither shapefile nor bounds are provided.

    """
    raster = rasterio.open(file)
    if shapefile is not None and bounds is None:
        out_array, out_meta = _clip_raster_with_shapefile(
            raster,
            shapefile,
            query,
            all_touched,
        )
    elif shapefile is None and bounds is not None:
        out_array, out_meta = _clip_raster_with_coords(
            raster,
            bounds,
        )
    elif shapefile is not None and bounds is not None:
        out_array, out_meta = _clip_raster_with_shapefile(
            raster,
            shapefile,
            query,
            all_touched,
        )
        out_array, out_meta = _clip_raster_with_coords(
            raster,
            bounds,
            intermediate_result=(out_array, out_meta),
        )
    else:
        raise ValueError("Either shapefile or bounds must be provided for clipping.")

    if input_folder is None:
        input_folder = file.parent

    if output_folder is not None:
        relative_file = file.relative_to(input_folder)
        out_file = output_folder / relative_file

        check_path(out_file.parent)
        save_raster(
            out_file,
            out_array,
            raster.crs,
            out_meta["height"],
            out_meta["width"],
            raster.nodata,
            out_meta["transform"],
        )
    return out_array, out_meta


def clip_raster_file(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    shapefile: Optional[Union[str, Path]] = None,
    query: Optional[str] = None,
    bounds: Optional[
        Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]
    ] = None,
    all_touched: bool = False,
):
    """
    Clips a raster file based on the provided parameters.

    Args:
        input_file (Union[str, Path]): The path to the input raster file.
        shapefile (Optional[Union[str, Path]]): The path to the shapefile used for clipping. Default is None.
        query (Optional[str]): An optional query string to filter the shapefile features. Default is None.
        bounds (Optional[Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]]):
            An optional tuple representing the bounding box coordinates (minx, miny, maxx, maxy) for clipping.
            Default is None.
        all_touched (bool): A flag indicating whether to include all pixels touched by the shapefile features.
            Default is False.

    Returns:
        Tuple: A tuple containing the clipped raster array and the metadata of the clipped raster.

    """
    input_file = Path(input_file)

    if shapefile:
        shapefile = Path(shapefile)

    out_array, out_meta = _clip_raster_process(
        file=input_file,
        input_folder=None,
        output_folder=None,
        shapefile=shapefile,
        query=query,
        bounds=bounds,
        all_touched=all_touched,
    )

    if output_file:
        output_file = Path(output_file)

        save_raster(
            output_file,
            out_array,
            out_meta["crs"],
            out_meta["height"],
            out_meta["width"],
            out_meta["nodata"],
            out_meta["transform"],
        )

    return out_array, out_meta


def clip_raster(
    input_folder: Union[str, Path],
    output_folder: Optional[Union[str, Path]],
    shapefile: Optional[Union[str, Path]],
    query: Optional[str],
    bounds: Optional[
        Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]
    ] = None,
    raster_extensions: List[str] = [".tif", ".tiff"],
    include_source: bool = True,
    recursive: bool = True,
    all_touched: bool = False,
    n_workers: int = mp.cpu_count(),
):
    """
    Clips rasters within the specified input folder to the extent of a shapefile or a bounding box.

    Args:
        input_folder (Union[str, Path]): Path to the input folder containing the rasters.
        output_folder (Optional[Union[str, Path]]): Path to the output folder where the clipped rasters will be saved.
        shapefile (Optional[Union[str, Path]]): Path to the shapefile used for clipping.
            If None, the rasters will be clipped to the specified bounding box.
        query (Optional[str]): Query string to filter the features in the shapefile.
            Only features that satisfy the query will be used for clipping. Ignored if shapefile is None.
        bounds (Optional[Tuple[Optional[Number], Optional[Number], Optional[Number], Optional[Number]]]):
            Bounding box coordinates (minx, miny, maxx, maxy) used for clipping. Ignored if shapefile is not None.
        raster_extensions (List[str]): List of file extensions to consider as rasters. Default is [".tif", ".tiff"].
        include_source (bool): Flag indicating whether to include the input folder itself as a source for clipping. Default is True.
        recursive (bool): Flag indicating whether to recursively search for rasters in subfolders. Default is True.
        all_touched (bool): Flag indicating whether to include all pixels touched by the shapefile or bounding box. Default is Fals.
        n_workers (int): Number of parallel workers to use for clipping. Default is the number of available CPU cores.
    """
    if recursive is False and include_source is False:
        raise ValueError(
            "Either recursive or include_source must be True to avoid an empty file list."
        )

    if recursive is True:
        folders, _ = create_file_folder_list(Path(input_folder))
    else:
        folders = []

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


def _unify_raster_grids(
    base_raster: Union[Path, rasterio.io.DatasetReader],
    raster_to_unify: Union[Path, rasterio.io.DatasetReader],
    resampling_method: Resampling,
    resampling_mode: str,
    same_extent: bool,
    same_shape: bool,
) -> Tuple[np.ndarray, dict]:

    if isinstance(base_raster, Path):
        base_raster = rasterio.open(base_raster)
    if isinstance(raster_to_unify, Path):
        raster_to_unify = rasterio.open(raster_to_unify)

    dst_crs = base_raster.crs
    dst_width = base_raster.width
    dst_height = base_raster.height
    dst_transform = base_raster.transform
    dst_resolution = (base_raster.transform.a, abs(base_raster.transform.e))

    out_meta = base_raster.meta.copy()
    raster = raster_to_unify

    if not same_extent:
        dst_transform, dst_width, dst_height = warp.calculate_default_transform(
            raster.crs,
            dst_crs,
            raster.width,
            raster.height,
            *raster.bounds,
            resolution=dst_resolution,
        )
        # Snap the corner coordinates to the grid
        x_distance_to_grid = dst_transform.c % dst_resolution[0]
        y_distance_to_grid = dst_transform.f % dst_resolution[1]

        if x_distance_to_grid > dst_resolution[0] / 2:  # Snap towards right
            c = dst_transform.c - x_distance_to_grid + dst_resolution[0]
        else:  # Snap towards left
            c = dst_transform.c - x_distance_to_grid

        if y_distance_to_grid > dst_resolution[1] / 2:  # Snap towards up
            f = dst_transform.f - y_distance_to_grid + dst_resolution[1]
        else:  # Snap towards bottom
            f = dst_transform.f - y_distance_to_grid

        # Create new transform with updated corner coordinates
        dst_transform = warp.Affine(
            dst_transform.a,  # Pixel size x
            dst_transform.b,  # Shear parameter
            c,  # Up-left corner x-coordinate
            dst_transform.d,  # Shear parameter
            dst_transform.e,  # Pixel size y
            f,  # Up-left corner y-coordinate
        )

        out_meta["transform"] = dst_transform
        out_meta["width"] = dst_width
        out_meta["height"] = dst_height

    dst_array = np.empty((base_raster.count, dst_height, dst_width))
    dst_array.fill(base_raster.nodata)

    src_array = raster.read()

    if resampling_mode == "auto":
        if np.issubdtype(src_array.dtype, np.integer):
            resampling_method = Resampling.nearest
        elif np.issubdtype(src_array.dtype, np.floating):
            resampling_method = Resampling.bilinear

    out_array = warp.reproject(
        source=src_array,
        src_crs=raster.crs,
        src_transform=raster.transform,
        src_nodata=raster.nodata,
        destination=dst_array,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        dst_nodata=base_raster.nodata,
        resampling=resampling_method,
    )[0]

    if same_shape is True:
        base_array = base_raster.read()
        out_array = np.where(
            base_array == base_raster.nodata, base_raster.nodata, out_array
        )

    return out_array, out_meta


def unify_raster_grids(
    base_raster: Union[str, Path],
    rasters_to_unify: Sequence[Union[str, Path]],
    resampling_method: Resampling = Resampling.nearest,
    resampling_mode: Literal["auto", "manual"] = "manual",
    same_extent: bool = False,
    same_shape: bool = False,
    n_workers: int = 1,
    verbose: int = 0,
) -> List[Tuple[np.ndarray, dict]]:
    """
    Unifies the grids of multiple rasters to match the grid of a base raster.

    Adapted core function from EIS Toolkit (main branch as of 29-01-2024).

    Args:
        base_raster (rasterio.io.DatasetReader): The base raster whose grid will be used as reference.
        rasters_to_unify (Sequence[rasterio.io.DatasetReader]): The rasters to be unified.
        resampling_method (Resampling, optional): The resampling method to be used. Defaults to Resampling.nearest.
        resampling_mode (str, optional): The resampling mode to be used. Defaults to "manual".
            If set to "auto", the resampling method will be determined automatically based on the data type.
            The manual set method will be overwritting in this case.
        same_extent (bool, optional): Whether to force all rasters to have the same extent as the base raster. Defaults to False.
        same_shape (bool, optional): Whether to force all rasters to have the same shape as the base raster. Defaults to False.

    Returns:
        List[Tuple[np.ndarray, dict]]: A list of tuples containing the unified rasters as numpy arrays and their associated metadata.
    """
    unified_results = []

    args_list = [
        (
            base_raster,
            file,
            resampling_method,
            resampling_mode,
            same_extent,
            same_shape,
        )
        for file in rasters_to_unify
    ]

    if n_workers > 1:
        if verbose == 1:
            print("Starting parallel processing...")

        with mp.Pool(n_workers) as pool:
            for result in pool.starmap(_unify_raster_grids, args_list):
                unified_results.append(result)
    else:
        for file in rasters_to_unify:
            out_array, out_meta = _unify_raster_grids(
                base_raster,
                file,
                resampling_method,
                resampling_mode,
                same_extent,
                same_shape,
            )
            unified_results.append((out_array, out_meta))

    if verbose == 1:
        print("Done!")

    return unified_results


# region: snap raster
def _return_raster_bounds(metadata: dict) -> rasterio.coords.BoundingBox:
    """
    Returns the bounds of a raster as a rasterio.coords.BoundingBox object.

    Args:
        metadata (dict): The metadata of the raster.

    Returns:
    rasterio.coords.BoundingBox: The bounds of the raster as a rasterio.coords.BoundingBox object.
    """
    raster_bounds = rasterio.transform.array_bounds(
        metadata["height"], metadata["width"], metadata["transform"]
    )
    return rasterio.coords.BoundingBox(*raster_bounds)


def _snap_raster(
    raster_array: np.ndarray,
    raster_meta: dict,
    snap_meta: dict,
) -> Tuple[np.ndarray, dict]:
    """
    Snaps a raster to align with the grid of a reference (snap) raster.

    This function adjusts the input raster so that its grid aligns with the grid of the snap raster.
    The snapping is done from the left-bottom corner to the nearest snap raster grid corner in the left-bottom direction.

    Args:
        raster_array (np.ndarray): The input raster array.
        raster_meta (dict): The metadata of the input raster.
        snap_data (dict): The dictionary containing the data for snapping the input raster.

    Returns:
        Tuple[np.ndarray, dict]:
            A tuple containing the snapped raster array and the updated metadata.
    """
    raster_bounds = _return_raster_bounds(raster_meta)
    snap_bounds = _return_raster_bounds(snap_meta)

    raster_px_size_x = abs(raster_meta["transform"].a)
    raster_px_size_y = abs(raster_meta["transform"].e)

    snap_px_size_x = abs(snap_meta["transform"].a)
    snap_px_size_y = abs(snap_meta["transform"].e)

    cells_added_x = math.ceil(snap_px_size_x / raster_px_size_x)
    cells_added_y = math.ceil(snap_px_size_y / raster_px_size_y)

    out_image = np.full(
        (
            raster_meta["count"],
            raster_meta["height"] + cells_added_y,
            raster_meta["width"] + cells_added_x,
        ),
        raster_meta["nodata"],
    )
    out_meta = raster_meta.copy()

    left_snap_coordinate = (
        snap_bounds.left
        + ((raster_bounds.left - snap_bounds.left) // snap_px_size_x) * snap_px_size_x
    )

    bottom_snap_coordinate = (
        snap_bounds.bottom
        + ((raster_bounds.bottom - snap_bounds.bottom) // snap_px_size_y)
        * snap_px_size_y
    )

    top_snap_coordinate = (
        bottom_snap_coordinate
        + (raster_meta["height"] + cells_added_y) * raster_px_size_y
    )

    x_distance = (raster_bounds.left - left_snap_coordinate) % raster_px_size_x
    x0 = int((raster_bounds.left - left_snap_coordinate) // raster_px_size_x)
    x1 = x0 + raster_meta["width"]

    y_distance = (raster_bounds.bottom - bottom_snap_coordinate) % raster_px_size_y
    y0 = int(
        cells_added_y
        - ((raster_bounds.bottom - bottom_snap_coordinate) // raster_px_size_y)
    )
    y1 = y0 + raster_meta["height"]

    if x_distance < raster_px_size_x / 2 and y_distance < raster_px_size_y / 2:
        out_image[:, y0:y1, x0:x1] = raster_array
    elif x_distance < raster_px_size_x / 2 and y_distance > raster_px_size_y / 2:
        out_image[:, y0 - 1 : y1 - 1, x0:x1] = raster_array
    elif x_distance > raster_px_size_x / 2 and y_distance > raster_px_size_y / 2:
        out_image[:, y0 - 1 : y1 - 1, x0 + 1 : x1 + 1] = raster_array
    else:
        out_image[:, y0:y1, x0 + 1 : x1 + 1] = raster_array

    out_transform = rasterio.Affine(
        raster_meta["transform"].a,
        raster_meta["transform"].b,
        left_snap_coordinate,
        raster_meta["transform"].d,
        raster_meta["transform"].e,
        top_snap_coordinate,
    )
    out_meta.update(
        {
            "transform": out_transform,
            "width": out_image.shape[-1],
            "height": out_image.shape[-2],
        }
    )

    return out_image, out_meta


#
def snap_raster(
    raster: Union[rasterio.DatasetReader, Tuple[np.ndarray, dict]],
    snap_to: Union[rasterio.DatasetReader, Tuple[Number, Number]],
) -> Tuple[np.ndarray, dict]:
    """
    Snaps/aligns raster to given snap raster.

    Raster is snapped from its left-bottom corner to nearest snap raster grid corner in left-bottom direction.
    If rasters are aligned, simply returns input raster data and metadata.

    Args:
        raster: The raster or an array with metadata to be snapped.
        snap_to:
            The reference raster or
            a rasterio metadata dictionary or
            a tuple containing the x and y origin to snap on.

    Returns:
        The snapped raster data.
        The updated metadata.

    Raises:
        ValueError: Raster and and snap raster are not in the same CRS.
    """
    if isinstance(raster, rasterio.DatasetReader):
        raster_array = raster.read()
        raster_meta = raster.meta.copy()
    else:
        raster_array = raster[0]
        raster_meta = raster[1]

    if isinstance(snap_to, rasterio.DatasetReader):
        snap_meta = snap_to.meta.copy()
    else:
        snap_meta = raster_meta.copy()
        snap_transform = rasterio.transform.from_origin(
            snap_to[0],
            snap_to[1],
            abs(snap_meta["transform"].a),
            abs(snap_meta["transform"].e),
        )
        snap_meta["transform"] = snap_transform

    src_bounds = _return_raster_bounds(raster_meta)
    snap_bounds = _return_raster_bounds(snap_meta)

    if not raster_meta["crs"] == snap_meta["crs"]:
        raise ValueError("Raster and and snap raster have different CRS.")

    if check_grid_alignment(src_bounds, snap_bounds) is True:
        raise ValueError("Raster grids are already aligned.")

    out_image, out_meta = _snap_raster(raster_array, raster_meta, snap_meta)
    return out_image, out_meta


# endregion: snap raster

# region: Test code

# endregion: Test code
