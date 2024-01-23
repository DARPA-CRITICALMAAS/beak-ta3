import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

import os
import multiprocessing as mp

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Sequence, List, Any
from numbers import Number
from collections import Counter
from tqdm import tqdm


def load_dataset(
    file: Union[Path, str],
    encoding_type: str = "ISO-8859-1",
    nrows: Optional[int] = None,
    **kwargs,
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
    folder: Path, extensions: List[str] = [".tif", ".tiff"], recursive: bool = False,
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

    if recursive is True:
        files = folder.rglob("*")
    else:
        files = folder.glob("*")
        
    for file in files:
        file = Path(file)
        if any(file.suffix.lower() == ext for ext in extensions):
            file_list.append(file)

    return file_list


def create_file_folder_list(
    root_folder: Path,
    exclude_types: Sequence[str] = [".aux", ".xml", ".cpg", ".ovr", ".dbf"],
) -> List[Path]:
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
                if not file_path.suffix.lower() in exclude_types:
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
        folder.mkdir(parents=True, exist_ok=True)

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
    """Convert dataframe to feather format.

    Args:
        data (pd.DataFrame): Input dataframe.
        file_path (Path): Output feather file.
    """
    threads = mp.cpu_count()
    chunksize = np.ceil(len(data) / threads).astype(int)
    data.to_feather(file_path, chunksize=chunksize)


def load_feather(
    file_path: Path,
    columns: Optional[List] = None,
    use_threads: bool = True,
    storage_options: Optional[dict] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load data from a feather file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the feather file.
        columns (Optional[List], optional): A list of column names to load. Defaults to None.
        use_threads (bool, optional): Whether to use multiple threads for reading. Defaults to True.
        storage_options (Optional[dict], optional): Additional options for storage. Defaults to None.
        nrows (Optional[int], optional): The number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    data = pd.read_feather(
        file_path,
        columns=columns,
        use_threads=use_threads,
        storage_options=storage_options,
    )

    if nrows:
        data = data[:nrows]
    return data


def spatial_filter(
    data: pd.DataFrame,
    longitude_column: Optional[str] = None,
    latitude_column: Optional[str] = None,
    longitude_min: Optional[Number] = None,
    longitude_max: Optional[Number] = None,
    latitude_min: Optional[Number] = None,
    latitude_max: Optional[Number] = None,
) -> pd.DataFrame:
    """Filter data by spatial extent.

    Args:
        data (pd.DataFrame): Input dataframe.
        longitude_min (Number): Minimum longitude.
        longitude_max (Number): Maximum longitude.
        latitude_min (Number): Minimum latitude.
        latitude_max (Number): Maximum latitude.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """

    # df_copy = df_copy[df_copy[COL_LONGITUDE] < max_long]
    if longitude_column is not None:
        data = (
            data[data[longitude_column] >= longitude_min]
            if longitude_min is not None
            else data
        )
        data = (
            data[data[longitude_column] <= longitude_max]
            if longitude_max is not None
            else data
        )
    if latitude_column is not None:
        data = (
            data[data[latitude_column] >= latitude_min]
            if latitude_min is not None
            else data
        )
        data = (
            data[data[latitude_column] <= latitude_max]
            if latitude_max is not None
            else data
        )

    return data


def load_model(
    model: dict,
    folders: Sequence[Path],
    file_extensions: Sequence[str] = [".tif", ".tiff"],
    verbose: int = 0,
):
    """Load model from dictionary and search for corresponding files.

    Args:
        model (dict): The model dictionary containing evidence layers.
        folders (Sequence[Path]): The list of folders to search for files.
        file_extensions (Sequence[str], optional): The file extensions to consider. Defaults to [".tif", ".tiff"].
        verbose (int, optional): The verbosity level. Defaults to 1.

    Raises:
        ValueError: Raised if no valid selection is found.
        ValueError: Raised if no files are found.

    Returns:
        dict: A dictionary containing the loaded model with corresponding files.
        list: A list of all files found.
        Counter: A counter containing the number of files for each evidence layer.
    """
    # Load evidence layers from model dictionary
    print("Loading model definition...")
    evidence_layers = []
    for layer, value in model.items():
        if value == True:
            evidence_layers.append(layer)

    if not evidence_layers:
        raise ValueError("No valid selection.")
    else:
        print(f"Selected {str(len(evidence_layers))} evidence layers.")
        if verbose == 1:
            [print(f"- {layer}") for layer in evidence_layers]

    # Create file list from provided folders
    file_list = []
    folder_list = []
    subfolders_list = []

    print("\nSearching for files and folders in provided paths...")
    for folder in folders:
        subfolders = create_file_folder_list(folder, exclude_types=[])[0]
        subfolders_list.extend(subfolders)

    folder_list = folders + subfolders_list
    for folder in folder_list:
        files = create_file_list(folder, file_extensions)
        file_list.extend(files)

    if not file_list:
        raise ValueError("No files found.")
    else:
        print(
            f"Found {str(len(folders))} folders, {str(len(subfolders_list))} subfolders and {str(len(file_list))} files."
        )

    # Create model file dictionary
    model_dict = {}
    for evidence_layer, value in model.items():
        if value is True:
            model_dict[evidence_layer] = None

    # Status
    print("\nSearching for corresponding files...")

    # Make comparison based on lower case names
    for layer in model_dict.keys():
        file_names_from_layers = []
        for extension in file_extensions:
            file_name_from_layer = layer + extension
            file_names_from_layers.append(file_name_from_layer.lower())

        matching_list = []
        for file in file_list:
            if str(file.name).lower() in file_names_from_layers:
                matching_list.append(file)

        if not matching_list:
            for folder in folder_list:
                if layer.lower() == str(folder.name).lower():
                    matching_list = create_file_list(folder, file_extensions)

        if matching_list:
            matching_list = sorted(matching_list, reverse=False)
            model_dict[layer] = matching_list

    for layer, files in model_dict.items():
        if files is None:
            print(f"Searched for '{layer}' but no matching file was found.")
        else:
            print(f"Found '{layer}' in {str(len(files))} file(s).")

    # Check if all layers have files
    print("\nEnsuring that all layers have matching files...")
    missing_layers = []
    for layer in model_dict.keys():
        if model_dict[layer] is None:
            missing_layers.append(layer)
            print(f"WARNING: No file found for evidence layer '{layer}'.")

    if missing_layers:
        print(f"WARNING: {str(len(missing_layers))} layers have no matching files.")
    else:
        print("All layers have matching files.")

    # Count the occurrences of each filename
    print("\nChecking files for multiple occurences...")
    file_list = [file for values in model_dict.values() for file in values]
    filename_counts = Counter([file.name for file in file_list])

    # Print the filenames that occur multiple times and their counts
    if max(filename_counts.values()) == 1:
        print("No duplicates found. All filenames occur only once.")
    else:
        if verbose == 1:
            for filename, count in filename_counts.items():
                if count > 1:
                    print(f"- '{filename}' occurs {count} times")
        else:
            print(
                f"Some filenames occur multiple times. Please check with option verbose=1 to see which files are affected."
            )

    return model_dict, file_list, filename_counts


def copy_folder_structure(
    source_folder: Path, destination_folder: Path, include_source=True, verbose: int = 0
):
    """Copy folder structure from source folder to destination folder.

    Args:
        source_folder (Path): The source folder.
        destination_folder (Path): The destination folder.
        verbose (int, optional): The verbosity level. Defaults to 1.
    """
    # Get all folders in the root folder
    folders, _ = create_file_folder_list(source_folder)
    
    if include_source is True:
        folders.insert(0, source_folder)
        
    if verbose == 1:
        print(f"Total of subfolders found: {len(folders)}")

    # Create folder structure
    for folder in folders:
        new_folder = folder.relative_to(source_folder)
        new_folder = destination_folder / new_folder
        check_path(new_folder)


# region: Test code
from importlib_resources import files

BASE_PATH = (
        files("beak.data")
        / "LAWLEY22-EXPORT"
        / "EPSG_4326_RES_0_05"
    )

source_folder = BASE_PATH / "COMPLETE_DATASET"
destination_folder = BASE_PATH / "COMPLETE_DATASET_COPY"

# copy_folder_structure(source_folder, destination_folder)

# endregion: Test code
