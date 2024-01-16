import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from utilities.preparation import create_encodings_from_dataframe, fill_nodata_with_mean
from utilities.raster_processing import check_path, save_raster
from utilities.misc import replace_invalid_characters
from rasterio import features, profiles, transform
from rasterio.crs import CRS
from rasterio.enums import MergeAlg
from shapely.wkt import loads
from tqdm import tqdm

# References
# Some non-trivial functionalities were adapted from other sources.
# The original sources are listed below and referenced in the code as well.
#
# EIS toolkit:
# GitHub repository https://github.com/GispoCoding/eis_toolkit under EUPL-1.2 license.


# region: General helper functions
def transform_from_geometries(
    geodataframe: gpd.GeoDataFrame, resolution: np.number
) -> Tuple[np.number, np.number, transform.Affine]:
    """
    Calculate the transform parameters required to convert the input geometries
    to a specified resolution. It takes a GeoDataFrame containing the geometries and the desired
    resolution as input.

    Adapted core function from EIS Toolkit (main branch as of 17-11-2023).

    Args:
        geodataframe (gpd.GeoDataFrame): The input GeoDataFrame containing the geometries.
        resolution (np.number): The desired resolution for the transformation.

    Returns:
        Tuple[float, float, transform.Affine]: A tuple containing the width, height, and transform
        parameters required for the transformation.

    """
    min_x, min_y, max_x, max_y = geodataframe.total_bounds

    out_width = int((max_x - min_x) / resolution)
    out_height = int((max_y - min_y) / resolution)

    out_transform = transform.from_origin(min_x, max_y, resolution, resolution)
    return out_width, out_height, out_transform

# endregion

# region: Convert to geodataframe
def create_geodataframe_from_polygons(
    data: pd.DataFrame,
    polygon_col: str,
    epsg_code: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a DataFrame containing polygon geometries.

    Args:
        data (pd.DataFrame): The input DataFrame containing the polygon data.
        polygon_col (str): The name of the column in the DataFrame that contains the polygon geometries.
        epsg_code (Optional[int]): The EPSG code specifying the coordinate reference system (CRS) of the geometries.
            If not provided, a ValueError will be raised.

    Returns:
        gpd.GeoDataFrame: The resulting GeoDataFrame with the polygon geometries.

    Raises:
        ValueError: If the `epsg_code` parameter is not provided.

    """
    if epsg_code is None:
        raise ValueError("Parameter epsg_code must be given.")

    data[polygon_col] = data[polygon_col].apply(loads)

    geodataframe = gpd.GeoDataFrame(
        data, geometry=polygon_col, crs=CRS.from_epsg(epsg_code)
    )
    
    if polygon_col != "geometry":
        geodataframe["geometry"] = geodataframe.geometry
        geodataframe.set_geometry("geometry", inplace=True)
        geodataframe.drop(columns=[polygon_col], inplace=True)
        
    return geodataframe


def create_geodataframe_from_points(
    data: pd.DataFrame,
    long_col: str,
    lat_col: str,
    epsg_code: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        long_col (str): The name of the column containing the longitude values.
        lat_col (str): The name of the column containing the latitude values.
        epsg_code (Optional[int]): The EPSG code specifying the coordinate reference system (CRS) of the data.
            If not provided, a ValueError will be raised.

    Returns:
        gpd.GeoDataFrame: The resulting GeoDataFrame.

    Raises:
        ValueError: If the epsg_code parameter is not provided.
    """
    if epsg_code is None:
        raise ValueError("Parameter epsg_code must be given.")

    geodataframe = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.loc[:, long_col], data.loc[:, lat_col]),
        crs=CRS.from_epsg(epsg_code),
    )
    return geodataframe


# endregion


# region: Raserize vector data
# Core functionality for creating raster data
def _rasterize_vector_helper(args):
    """
    Pass arguments to rasterize_vector_process.

    Args:
        args: Tuple of arguments to be passed to rasterize_vector_process.

    Returns:
        The result of the rasterize_vector_process function.
    """
    return _rasterize_vector_process(*args)


# Helper function to func partial process
def _rasterize_vector_process(
    value_column: str,
    values: np.ndarray,
    geometries: gpd.array.GeometryArray,
    height: int,
    width: int,
    nodata_value: np.number,
    transform: transform.Affine,
    all_touched: bool,
    merge_strategy: str,
    default_value: np.number,
    dtype: Optional[np.dtype],
    impute_nodata: bool,
):
    """
    Rasterize vector data based on the provided parameters.

    Args:
        value_column (str): The name of the column containing the values to be rasterized.
        values (np.ndarray): The array of values to be rasterized.
        geometries (gpd.array.GeometryArray): The array of geometries to be rasterized.
        height (int): The height of the output raster.
        width (int): The width of the output raster.
        nodata_value (np.number): The nodata value to be used in the output raster.
        transform (transform.Affine): The affine transformation to be applied to the output raster.
        all_touched (bool): Whether to consider all pixels touched by the geometries as valid.
        merge_strategy (str): The merge strategy to be used when multiple geometries overlap a pixel.
        default_value (np.number): The default value to be used for pixels without any geometries.
        dtype (Optional[np.dtype]): The data type of the output raster. If None, it will be inferred from the values array.
        impute_nodata (bool): Whether to impute single nodata values in the output raster.

    Returns:
        Tuple[str, np.ndarray, transform.Affine]: A tuple containing the name of the output column, the rasterized array, and the output transformation.
    """
    # Create geometry-value pairs
    geometry_value_pairs = list(zip(geometries, values))
    dtype = values.dtype if dtype is None else dtype

    # Rasterize
    out_array = features.rasterize(
        shapes=geometry_value_pairs,
        out_shape=(height, width),
        fill=nodata_value,
        transform=transform,
        all_touched=all_touched,
        merge_alg=getattr(MergeAlg, merge_strategy),
        default_value=default_value,
    )

    # Impute single nodata values
    if impute_nodata == True:
        out_array = fill_nodata_with_mean(
            array=out_array,
            nodata_value=nodata_value,
        )

    # Prepare output
    out_column = value_column
    out_array = out_array.reshape(1, out_array.shape[0], out_array.shape[1])
    out_transform = transform

    return out_column, out_array.astype(dtype), out_transform


# Rasterize vector main function
def rasterize_vector(
    value_type: Literal["categorical", "numerical", "ground_truth"],
    value_columns: List[str],
    geodataframe: gpd.GeoDataFrame,
    default_value: np.number = 1,
    nodata_value: np.number = -99999,
    resolution: Optional[np.number] = None,
    epsg_code: Optional[int] = None,
    base_raster_profile: Optional[Union[profiles.Profile, dict]] = None,
    merge_strategy: str = "replace",
    all_touched: bool = True,
    dtype: Optional[np.dtype] = None,
    impute_nodata: bool = False,
    export_absent: bool = False,
    raster_save: bool = False,
    raster_save_folder: Optional[Path] = None,
    n_workers: int = mp.cpu_count(),
    chunksize: Optional[int] = None,
) -> Tuple[List, List, List]:
    """
    Rasterize vector data.

    Args:
        value_type (Literal["categorical", "numerical", "ground_truth"]): The type of the values to be rasterized.
        value_columns (List[str]): The columns containing the values to be rasterized.
        geodataframe (gpd.GeoDataFrame): The GeoDataFrame containing the vector data.
        default_value (np.number): The default value to be assigned to raster cells without a value. Defaults to 1.
        nodata_value (np.number): The nodata value to be assigned to raster cells. Defaults to -99999.
        resolution (Optional[np.number]): The resolution of the raster cells. Defaults to None.
        epsg_code (Optional[int]): The EPSG code of the coordinate reference system. Defaults to None.
        base_raster_profile (Optional[Union[profiles.Profile, dict]]): The base raster profile to use for rasterization. Defaults to None.
        merge_strategy (str): The strategy to use when merging rasterized values. Defaults to "replace".
        all_touched (bool): Whether to consider all pixels touched by the vector geometry. Defaults to True.
        dtype (Optional[np.dtype]): The data type of the raster cells. Defaults to None.
        impute_nodata (bool): Whether to impute nodata values in the raster. Defaults to False.
        export_absent (bool): Whether to export absent values as separate columns. Defaults to False.
        raster_save (bool): Whether to save the rasterized values as raster files. Defaults to False.
        raster_save_folder (Path): The folder to save the raster files. Defaults to None.
        n_workers (int): The number of worker processes to use for parallel rasterization. Defaults to mp.cpu_count().
        chunksize (Optional[int]): The number of value columns to process in each worker process. Defaults to None.

    Returns:
        Tuple[List, List, List]: A tuple containing the list of output column names, the list of output rasters, and the list of output transforms.
    """
    # Check input arguments
    if len(value_columns) == 0:
        return [], [], []

    if resolution is not None and base_raster_profile is not None:
        raise ValueError(
            "Provide either resolution or base_raster_profile, but not both."
        )

    if raster_save == True and raster_save_folder is None:
        raise ValueError("Expected raster_save_folder to be given.")

    # Special actions for categorical and ground_truth data
    if value_type == "categorical" or value_type == "ground_truth":
        # Create binary encodings
        data_encoded, value_columns = create_encodings_from_dataframe(
            value_columns, geodataframe, export_absent
        )

        if value_type == "ground_truth":
            # Check if ground truth columns are present
            if len(data_encoded.columns) == 0:
                raise ValueError("No ground truth columns found.")

        # Append coordinates
        dataframe = pd.concat([data_encoded, geodataframe.geometry], axis=1)

        # Re-create GeoDataFrame since concat removes the geo-character
        geodataframe = gpd.GeoDataFrame(
            dataframe, geometry=geodataframe.geometry, crs=CRS.from_epsg(epsg_code)
        )

    # Create Affine.transform
    geometries = geodataframe.geometry.values
    width, height, transform = (
        transform_from_geometries(geodataframe, resolution)
        if resolution is not None
        else (
            base_raster_profile["width"],
            base_raster_profile["height"],
            base_raster_profile["transform"],
        )
    )

    # Set arguments for rasterization
    count = len(value_columns)
    args = zip(
        value_columns,
        [geodataframe[column].values for column in value_columns],
        [geometries] * count,
        [height] * count,
        [width] * count,
        [nodata_value] * count,
        [transform] * count,
        [all_touched] * count,
        [merge_strategy] * count,
        [default_value] * count,
        [dtype] * count,
        [impute_nodata] * count,
    )

    # Initialize results list
    out_columns = []
    out_rasters = []
    out_transforms = []

    # Set up multiprocessing
    pool = mp.Pool(n_workers)

    # Show number of threads
    print(f"Number of threads rasterizing: {n_workers}")

    # Set chunksize
    if chunksize is None:
        chunksize = (
            1
            if len(value_columns) < n_workers
            else int(np.ceil(len(value_columns) / n_workers))
        )

    # Rasterize and save in parallel
    with tqdm(total=len(value_columns), desc="Rasterizing") as pbar:
        for result in pool.imap_unordered(
            _rasterize_vector_helper, args, chunksize=chunksize
        ):
            # Unpack result
            column, raster, transform = result

            # Create result lists
            out_columns.append(column)
            out_rasters.append(raster)
            out_transforms.append(transform)

            # Check column names and reduce underscores
            column = replace_invalid_characters(column)
            check_path(raster_save_folder)

            if raster_save == True:
                save_raster(
                    path=Path(raster_save_folder) / f"{column}.tif",
                    array=raster,
                    nodata_value=nodata_value,
                    epsg_code=epsg_code,
                    height=raster.shape[1],
                    width=raster.shape[2],
                    transform=transform,
                )

            # Short wait
            pbar.update(1)
            time.sleep(0.1)

    return out_columns, out_rasters, out_transforms


# endregion
