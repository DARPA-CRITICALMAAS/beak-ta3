import numpy as np
import rasterio

from pathlib import Path
from typing import Union, Optional

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
