import os
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import multiprocessing as mp
import rasterio
from rasterio.crs import CRS
from tqdm import tqdm


def load_dataset(
    file: Path, encoding_type: str = "ISO-8859-1", nrows: Optional[int] = None, **kwargs
) -> pd.DataFrame:
    """
    Load a text-based dataset from disk.

    Args:
        file (Path): Location of the file to be loaded.
        encoding_type (str): Text encoding of the input file. Defaults to "ISO-8859-1".
        nrows (int, optional): Number of rows to be loaded. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to pd.read_csv().

    Returns:
        pd.DataFrame: Table with loaded data.
    """
    return pd.read_csv(file, encoding=encoding_type, nrows=nrows, **kwargs)


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
            if dirs:
                file_list.append(Path(root) / folder / file)
            else:
                file_path = Path(root) / file

                # Ignore metadata files
                if not file_path.suffix.lower() in [
                    ".aux",
                    ".xml",
                    ".cpg",
                    ".ovr",
                    ".dbf",
                ]:
                    file_list.append(file_path)

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

    return folder


def save_raster(
    path: Path,
    array: np.ndarray,
    epsg_code: int,
    height: int,
    width: int,
    nodata_value: np.number,
    transform: Any,
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
        transform (affine.Affine): The affine transformation matrix that maps pixel coordinates to CRS coordinates.

    Returns:
        (None): None
    """
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)

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


def dataframe_to_feather(data: pd.DataFrame, file_path: Path):
    threads = mp.cpu_count()
    chunksize = np.ceil(len(data) / threads).astype(int)
    data.to_feather(file_path, chunksize=chunksize)


def load_feather(
    file_path: Path,
    columns: Optional[List] = None,
    use_threads: bool = True,
    storage_options: Optional[dict] = None,
) -> pd.DataFrame:
    return pd.read_feather(
        file_path,
        columns=columns,
        use_threads=use_threads,
        storage_options=storage_options,
    )
