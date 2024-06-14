import multiprocessing as mp
import os

from pathlib import Path
from typing import Optional, Tuple, Union, Literal

import numpy as np
import geopandas as gpd
import rasterio
import pyproj

from rasterio.crs import CRS
from tqdm import tqdm


# region: reproject vector data
def _reproject_vector_data(
    data: Union[Path, str, gpd.GeoDataFrame],
    encoding: str = "utf-8",
    epsg: Optional[int] = None,
    crs: Optional[Union[str, rasterio.crs.CRS, pyproj.crs.CRS]] = None,
    query: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Reprojects a vector file (shapefile or geopackage) to a new coordinate reference system (CRS).

    Parameters:
    file (Union[Path, str]): The path to the input vector file.
    epsg (Optional[int]): The EPSG code of the target CRS.
    crs (Optional[str]): The CRS string of the target CRS.

    Returns:
    gpd.GeoDataFrame: The reprojected vector data as a GeoDataFrame.

    Raises:
    ValueError: If the input file is not a shapefile or geopackage.
    ValueError: If neither epsg nor crs is specified.
    ValueError: If both epsg and crs are specified.
    """
    if epsg is None and crs is None:
        raise ValueError(f"Must specify either epsg or crs.")
    elif epsg is not None and crs is not None:
        raise ValueError(f"Cannot specify both epsg and crs.")

    if isinstance(data, str):
        file = Path(file)

    if isinstance(data, gpd.GeoDataFrame):
        gdf = data
    else:
        gdf = gpd.read_file(data, encoding=encoding)

    if query is not None:
        gdf = gdf.query(query)

    out_gdf = gdf.to_crs(crs=crs, epsg=epsg, inplace=False)
    return out_gdf


# endregion: reproject vector data
