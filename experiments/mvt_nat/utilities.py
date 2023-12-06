import os
import numpy as np
import pandas as pd
import scipy
import rasterio
import geopandas as gpd
import multiprocessing as mp
import warnings
import re
import time

from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Tuple, Union, Literal
from sklearn.preprocessing import StandardScaler
from rasterio import features, transform, profiles
from rasterio.enums import MergeAlg
from rasterio.crs import CRS
from rasterio import warp
from shapely.wkt import loads

# References
# Some non-trivial functionalities were adapted from other sources.
# The original sources are listed below and referenced in the code as well.
#
# EIS toolkit:
# GitHub repository https://github.com/GispoCoding/eis_toolkit under EUPL-1.2 license.


# General helper functions
def replace_invalid_characters(string: str) -> str:
    """
    Replace invalid characters in a string with an underscore.
    Multiple consecutive underscores will be cropped to one.

    Args:
        string (str): Input string.

    Returns:
        str: String with replaced characters.
    """
    string = re.sub(r"[ /().,]", "_", string)
    string = re.sub(r"(_+)", "_", string)
    return string


# IO functions
def load_dataset(
    file: Path, encoding_type: str = "ISO-8859-1", nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a text-based dataset from disk.

    Args:
        file (Path): Location of the file to be loaded.
        encoding_type (str): Text encoding of the input file. Defaults to "ISO-8859-1".
        nrows (Optional[int]): Number of rows to be loaded. Defaults to None.

    Returns:
        pd.DataFrame: Table with loaded data.
    """
    return pd.read_csv(file, encoding=encoding_type, nrows=nrows)


def load_raster(file: Path) -> rasterio.io.DatasetReader:
    """
    Load a single raster file using rasterio.

    Args:
        file (Path): The path to the raster file.

    Returns:
        rasterio.io.DatasetReader: The opened raster dataset.

    """
    return rasterio.open(file)


def load_rasters(
    folder: Path, extensions: List[str] = [".tif", ".tiff"]
) -> List[rasterio.io.DatasetReader]:
    """
    Load raster files of a given type from a folder.

    Args:
        folder (Path): The folder path where the raster files are located.
        extensions (List[str]): List of file extensions to consider. Defaults to [".tif", ".tiff"].

    Returns:
        Tuple[List[Path], List[rasterio.io.DatasetReader]]: A tuple containing the list of file paths and the loaded raster datasets.
    """
    file_list = create_file_list(folder, extensions)
    loaded_rasters = []

    for file in tqdm(file_list):
        loaded_raster = load_raster(file)
        loaded_rasters.append(loaded_raster)

    return file_list, loaded_rasters


def read_raster(raster: rasterio.io.DatasetReader) -> np.ndarray:
    """
    Read a raster single-band raster dataset.

    Args:
        raster (rasterio.io.DatasetReader): The raster dataset to read.

    Returns:
        np.ndarray: The raster data as a NumPy array.
    """
    assert raster.band_count == 1
    return raster.read()


def read_rasters(raster_list: List[rasterio.io.DatasetReader]) -> List[np.ndarray]:
    """
    Read multiple rasters and return them as a list of NumPy arrays.

    Args:
        raster_list (List[rasterio.io.DatasetReader]): A list of rasterio dataset readers.

    Returns:
        List[np.ndarray]: A list of NumPy arrays representing the read rasters.
    """
    return [read_raster(raster) for raster in raster_list]


def get_filename_and_extension(file: Path) -> Tuple[str, str]:
    """
    Get the filename and extension from a given file path.

    Args:
        file (Path): The path to the file.

    Returns:
        Tuple[str, str]: A tuple containing the filename and extension.

    """
    return file.stem, file.suffix


def create_file_list(
    folder: Path, extensions: List[str] = [".tif", ".tiff"]
) -> List[Path]:
    """
    Create a list of files in the specified folder with the given extensions.

    Args:
        folder (Path): The folder path to search for files.
        extensions (List[str]): The list of file extensions to include. Defaults to [".tif", ".tiff"].

    Returns:
        List[Path]: A list of Path objects representing the files found.
    """
    file_list = []

    for file in folder.glob("*"):
        file = Path(file)
        if any(file.suffix.lower() == ext for ext in extensions):
            file_list.append(file)

    return file_list


def create_file_folder_list(root_folder: Path) -> List[Path]:
    """
    Create a list of files and folders in a root folder (including subfolders).

    Args:
        root_folder (Path): The root folder to search for folders.

    Returns:
        Tuple[List[Path], List[Path]]: A tuple containing two lists - the list of folders and the list of files.
    """
    folder_list = []
    file_list = []

    for root, dirs, files in os.walk(root_folder):
        for folder in dirs:
            folder_list.append(Path(root) / folder)

        for file in files:
            file_list.append(Path(root) / folder / file)

    return folder_list, file_list


def check_path(folder: Path):
    """
    Check if path exists and create if not.

    Args:
        folder (Path): The path to check and create if necessary.

    Returns:
        (None): None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_raster(
    path: Path,
    array: np.ndarray,
    epsg_code: int,
    height: int,
    width: int,
    nodata_value: np.number,
    transform: dict,
):
    """
    Save raster data to disk.

    Args:
        path (Path): The file path where the raster will be saved.
        array (np.ndarray): The raster data as a NumPy array.
        epsg_code (int): The EPSG code specifying the coordinate reference system (CRS) of the raster.
        height (int): The height (number of rows) of the raster.
        width (int): The width (number of columns) of the raster.
        nodata_value (np.number): The nodata value of the raster.
        transform (dict): The affine transformation matrix that maps pixel coordinates to CRS coordinates.

    Returns:
        (None): None
    """
    count = array.shape[0]
    dtype = array.dtype

    meta = {
        "driver": "GTiff",
        "dtype": str(dtype),
        "nodata": nodata_value,
        "width": width,
        "height": height,
        "count": count,
        "crs": CRS.from_epsg(epsg_code),
        "transform": transform,
    }

    with rasterio.open(path, "w", **meta) as dst:
        for i in range(0, count):
            dst.write(array[i].astype(dtype), i + 1)

        dst.close()


def update_raster_metadata(raster: rasterio.io.DatasetReader, **kwargs) -> dict:
    """
    Update raster metadata.

    Args:
        raster (rasterio.io.DatasetReader): The input raster dataset.
        **kwargs (tuple): Additional metadata key-value pairs to update.

    Returns:
        dict: The updated metadata dictionary.

    """
    meta = raster.meta.copy()
    meta.update(kwargs)
    return meta


# Statistics related functions
def get_outliers_zscore(
    data: pd.DataFrame, column: str, threshold: np.number = 3.0
) -> pd.DataFrame:
    """
    Get outliers based on the z-score using scikit-learn.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to calculate z-scores and identify outliers.
        threshold (np.number): The threshold value for identifying outliers. Defaults to 3.0.

    Returns:
        pd.DataFrame: A DataFrame containing the outliers based on the z-score.

    """
    # Extract the column data
    column_data = data[[column]]

    # Use StandardScaler to calculate z-scores
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(column_data)

    # Identify outliers and return as DataFrame
    outliers = pd.DataFrame(data.loc[np.abs(z_scores) > threshold, column])

    return outliers


def get_outliers_iqr(
    data: pd.DataFrame, column: str, threshold: np.number = 1.5
) -> pd.Series:
    """
    Get outliers based on the IQR method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to calculate outliers for.
        threshold (np.number): The threshold value for outlier detection. Defaults to 1.5.

    Returns:
        pd.Series: A Series containing the outliers.

    """
    # Extract the column data
    column_data = data[column]

    # Calculate IQR
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Identify outliers
    outliers = pd.DataFrame(
        data.loc[(column_data < lower_bound) | (column_data > upper_bound), column]
    )
    return outliers


# Vector processing related functions
def buffer_vector(
    geodataframe: gpd.GeoDataFrame, buffer_value: np.number
) -> gpd.GeoDataFrame:
    """
    Buffer vector data.
    Adapted core function from EIS Toolkit (main branch as of 17-11-2023).

    Args:
        geodataframe (gpd.GeoDataFrame): The input GeoDataFrame.
        buffer_value (np.number): The buffer distance.

    Returns:
        gpd.GeoDataFrame: The buffered GeoDataFrame.
    """
    geodataframe = geodataframe.copy()
    geodataframe["geometry"] = geodataframe["geometry"].apply(
        lambda geom: geom.buffer(buffer_value)
    )
    return geodataframe


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
    geodataframe["geometry"] = geodataframe.geometry
    geodataframe.set_geometry("geometry", inplace=True)

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


# Raster related functions
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


def create_encodings_from_dataframe(
    value_columns: List[str],
    data: pd.DataFrame,
    export_absent: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create encodings for categorical data.

    Args:
        value_columns (List[str]): List of column names containing categorical values.
        data (pd.DataFrame): Input data frame.
        export_absent (bool): Flag indicating whether to export "Absent" columns.

    Returns:
        pd.DataFrame: Encoded data frame.
        List[str]: List of new column names.

    """
    # Create binary encodings
    data_encoded = pd.get_dummies(data[value_columns], prefix=value_columns).astype(
        np.uint8
    )

    # Get new column names
    new_value_columns = data_encoded.columns.to_list()

    if export_absent == False:
        # Get rid of the "Absent" columns since they are not needed in binary encoding
        new_value_columns = [
            column for column in new_value_columns if "Absent" not in column
        ]
        data_encoded = data_encoded[new_value_columns]

    return data_encoded, new_value_columns


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
