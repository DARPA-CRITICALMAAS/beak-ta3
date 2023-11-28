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


def load_dataset(
    file: Path, encoding_type: str = "ISO-8859-1", nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load text-based dataset."""
    return pd.read_csv(file, encoding=encoding_type, nrows=nrows)


def load_raster(file: Path) -> rasterio.io.DatasetReader:
    """Load raster dataset."""
    return rasterio.open(file)


def read_raster(raster: rasterio.io.DatasetReader) -> np.ndarray:
    """Read raster dataset. Accepts only single-band rasters for simplicity."""
    assert raster.band_count == 1
    return raster.read()


def create_file_list(folder: Path, extensions: List[str]) -> List[Path]:
    """Create a list of files with given extensions."""
    file_list = []

    for file in folder.glob("*"):
        file = Path(file)
        if any(file.suffix.lower() == ext for ext in extensions):
            file_list.append(file)

    return file_list


def get_filename_and_extension(file: Path) -> Tuple[str, str]:
    """Get filename and extension from file."""
    return file.stem, file.suffix


def create_folder_list(root_folder: Path) -> List[Path]:
    """Create a list of folders in a root folder (including subfolders)."""
    folder_list = []
    file_list = []

    for root, dirs, files in os.walk(root_folder):
        for folder in dirs:
            folder_list.append(Path(root) / folder)

        for file in files:
            file_list.append(Path(root) / folder / file)

    return folder_list, file_list


def load_rasters(
    folder: Path, extensions: List = [".tif", ".tiff"]
) -> List[rasterio.io.DatasetReader]:
    """Load multiple raster datasets."""
    file_list = create_file_list(folder, extensions)
    loaded_rasters = []

    for file in tqdm(file_list):
        loaded_raster = load_raster(file)
        loaded_rasters.append(loaded_raster)

    return file_list, loaded_rasters


def read_rasters(raster_list: List[rasterio.io.DatasetReader]) -> List[np.ndarray]:
    """Read multiple raster datasets."""
    return [read_raster(raster) for raster in raster_list]


def save_raster(
    path: Path,
    array: np.ndarray,
    epsg_code: int,
    height: int,
    width: int,
    nodata_value: np.number,
    transform: dict,
):
    """Save raster data to disk."""
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
    """Update raster metadata."""
    meta = raster.meta.copy()
    meta.update(kwargs)
    return meta


def check_path(folder: Path):
    """Check if path exists and create if not."""
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_outliers_zscore(
    data: pd.DataFrame, column: str, threshold: float = 3.0
) -> pd.DataFrame:
    """Get outliers based on the z-score using scikit-learn."""

    # Extract the column data
    column_data = data[[column]]

    # Use StandardScaler to calculate z-scores
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(column_data)

    # Identify outliers and return as DataFrame
    outliers = pd.DataFrame(data.loc[np.abs(z_scores) > threshold, column])

    return outliers


def get_outliers_iqr(
    data: pd.DataFrame, column: str, threshold: float = 1.5
) -> pd.Series:
    """Get outliers based on the IQR method."""

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


# Adapted core function from EIS Toolkit (main branch as of 2023-11-17)
def buffer_vector(
    geodataframe: gpd.GeoDataFrame, buffer_value: np.number
) -> gpd.GeoDataFrame:
    """Buffer vector data."""
    geodataframe = geodataframe.copy()
    geodataframe["geometry"] = geodataframe["geometry"].apply(
        lambda geom: geom.buffer(buffer_value)
    )
    return geodataframe


# Adapted core function from EIS Toolkit (main branch as of 2023-11-17)
def _transform_from_geometries(
    geodataframe: gpd.GeoDataFrame, resolution: np.number
) -> Tuple[float, float, transform.Affine]:
    """Determine transform from the input geometries."""
    min_x, min_y, max_x, max_y = geodataframe.total_bounds

    out_width = int((max_x - min_x) / resolution)
    out_height = int((max_y - min_y) / resolution)

    out_transform = transform.from_origin(min_x, max_y, resolution, resolution)
    return out_width, out_height, out_transform


def _create_geodataframe_from_polygons(
    data: pd.DataFrame,
    polygon_col: str,
    epsg_code: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from DataFrame."""
    if epsg_code is None:
        raise ValueError("Parameter epsg_code must be given.")

    data[polygon_col] = data[polygon_col].apply(loads)

    geodataframe = gpd.GeoDataFrame(
        data, geometry=polygon_col, crs=CRS.from_epsg(epsg_code)
    )
    geodataframe["geometry"] = geodataframe.geometry
    geodataframe.set_geometry("geometry", inplace=True)

    return geodataframe


def _create_geodataframe_from_points(
    data: pd.DataFrame,
    long_col: str,
    lat_col: str,
    epsg_code: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from DataFrame."""
    if epsg_code is None:
        raise ValueError("Parameter epsg_code must be given.")

    geodataframe = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.loc[:, long_col], data.loc[:, lat_col]),
        crs=CRS.from_epsg(epsg_code),
    )
    return geodataframe


def _replace_invalid_chars(string: str) -> str:
    """Replace invalid characters in string."""
    string = re.sub(r"[ /().,]", "_", string)
    string = re.sub(r"(_+)", "_", string)
    return string


def fill_nodata_with_mean(
    array: np.ndarray,
    nodata_value: np.number,
    size: Optional[int] = 3,
    num_nan_max: Optional[int] = 4,
) -> np.ndarray:
    """Fill nodata values with mean from sourrounding cells."""
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


# Helper function to pass arguments
def _rasterize_vector_helper(args):
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


def _rasterize_vector_create_encodings(
    value_columns: List[str],
    data: pd.DataFrame,
    export_absent,
) -> pd.DataFrame:
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
    """Rasterize vector data."""
    # Check input arguments
    if resolution is not None and base_raster_profile is not None:
        raise ValueError(
            "Provide either resolution or base_raster_profile, but not both."
        )

    if raster_save == True and raster_save_folder is None:
        raise ValueError("Expected raster_save_folder to be given.")

    # Special actions for categorical and ground_truth data
    if value_type == "categorical" or value_type == "ground_truth":
        # Create binary encodings
        data_encoded, value_columns = _rasterize_vector_create_encodings(
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
        _transform_from_geometries(geodataframe, resolution)
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
            column = _replace_invalid_chars(column)
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


# Reproject raster: Adapted function from EIS Toolkit (main branch as of 2023-11-17)
def reproject_raster(
    raster: rasterio.io.DatasetReader,
    target_crs: int,
    target_resolution: Optional[np.number] = None,
    resampling_method: warp.Resampling = warp.Resampling.nearest,
) -> Tuple[np.ndarray, dict]:
    """Reproject raster data to a new coordinate reference system."""
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
