import rasterio
import numpy as np

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Sequence


def check_positives(file_path: Union[str, Path], band: Optional[int] = None):
    """
    Check the number of positive values in a raster.

    Args:
      file_path (Union[str, Path]): The path to the raster file.
      band (Optional[int]): The band number to read from the raster. If not specified, the first band will be used.

    Returns:
      int: The number of positive values in the raster.
    """
    raster = rasterio.open(file_path)
    band = 1 if band is None else band
    array = raster.read(band)
    array = np.where(array == raster.nodata, np.nan, array)
    return np.nansum(array)


def check_write_permissions(path: Union[Path, str], overwrite: bool = False) -> bool:
    """
    Check if a file exists and if it should be overwritten.

    Args:
      path (Union[Path, str]): The path to the file.
      overwrite (bool): Whether to overwrite the file if it exists. Defaults to False.

    Returns:
        bool: True if the file should be overwritten or does not exist, False otherwise.
    """
    path = Path(path)

    if path.exists():
        if overwrite is False:
            print(f"File already exists: {path.name}.")
            return False
        elif overwrite is True:
            print(f"Overwriting file: {path.name}.")
            return True
    else:
        return True
  
  
def check_grid_alignment(
    src_bounds: rasterio.coords.BoundingBox, target_bounds: rasterio.coords.BoundingBox
) -> bool:
    """
    Checks if two bounding boxes are aligned.

    Args:
        src_bounds (rasterio.coords.BoundingBox): The source bounding box.
        target_bounds (rasterio.coords.BoundingBox): The target bounding box.

    Returns:
        bool: Whether the bounding boxes are aligned.
    """
    if (
        src_bounds.left == target_bounds.left
        and src_bounds.bottom == target_bounds.bottom
    ):
        return True
    else:
        return False


def _check_raster_dim(array: np.ndarray):
    """
    Check if the raster array has the expected dimensions.

    Args:
        array (np.ndarray): The raster array.

    Raise:
        valueError: If the raster array does not have the expected dimensions (2D or 3D).
    """
    if not (array.ndim == 2 or array.ndim == 3):
        raise ValueError("Invalid raster array dimensions. Expected 2D or 3D array.")


def check_raster_input(
    path: Optional[Union[str, Path]] = None,
    raster: Optional[rasterio.io.DatasetReader] = None,
    data: Optional[Tuple[np.ndarray, Dict]] = None,
    bands: Optional[Union[int, Sequence[int]]] = None,
):
    """
    Check the size and dimensions of a raster array.

    This function verifies that the raster array has the expected dimensions (2D or 3D)
        and optionally checks if a specified band exists.

    Args:
        path (Optional[Union[str, Path]]): The path to the raster file.
        raster (Optional[rasterio.io.DatasetReader]): An open rasterio dataset.
        data (Optional[Tuple[np.ndarray, Dict]]): A tuple containing the raster array and its metadata.
        bands (Optional[Union[int, Sequence[int]]]): The band number(s) to check.
            If specified, the function will verify that the raster array contains this band(s).
            If a 2D array was provided, the band selection will be ignored.

        Only one of `path`, `raster`, or `data` should be provided.

    Returns:
        Tuple[np.ndarray, Dict]: A tuple containing the raster array and its metadata.

    Raises:
        ValueError: If more than one of `path`, `raster`, or `data` is provided.
        ValueError: If the raster array does not have the expected dimensions (2D or 3D).
        ValueError: If the specified band number does not exist in the raster array.
    """

    # Inputs
    counter = sum(value is not None for value in [path, raster, data])

    if counter == 0:
        raise ValueError("No valid input provided.")
    if counter > 1:
        raise ValueError("Only one of path, raster, or data should be provided.")

    # Array dimensions and size
    if data is None:
        raster = rasterio.open(path) if isinstance(path, (str, Path)) else raster

        array = raster.read()
        meta = raster.meta.copy()
    else:
        array, meta = data

    if bands is not None:
        _check_raster_bands(array, bands)
    else:
        _check_raster_dim(array)

    return array, meta


def _check_raster_bands(array: np.ndarray, bands: Union[int, Sequence[int]]):
    """
    Check if the raster array has the sufficient number of bands for the specified band number.

    Args:
        array (np.ndarray): The 2D or 3D raster array.
        bands (Union[int, Sequence[int]]): The band number(s) to check.

    Raises:
        valueError: If the raster array has not the specified band.
    """
    # Raster dimensions
    _check_raster_dim(array)

    # Band selection
    array = np.expand_dims(array, 0) if array.ndim == 2 else array
    bands = [bands] if isinstance(bands, int) else bands

    if not all(1 <= band <= array.shape[0] for band in bands):
        raise ValueError(f"Invalid band selection.")
