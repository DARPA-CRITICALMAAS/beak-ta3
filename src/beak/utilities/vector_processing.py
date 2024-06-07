import multiprocessing as mp
import os

from pathlib import Path
from typing import Optional, Tuple, Union, Literal

import numpy as np
import geopandas as gpd
import rasterio

from rasterio import warp
from rasterio.crs import CRS
from tqdm import tqdm


# region: reproject vector data
def _reproject_vector_data(
    file: Union[Path, str],
    epsg: Optional[int] = None,
    crs: Optional[str] = None,
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

    file = Path(file)

    if not file.suffix == ".shp" or file.suffix == ".gpkg":
        raise ValueError(
            f"Input file must be a shapefile (.shp) or geopackage (.gpkg)."
        )

    if epsg is None and crs is None:
        raise ValueError(f"Must specify either epsg or crs.")
    elif epsg is not None and crs is not None:
        raise ValueError(f"Cannot specify both epsg and crs.")

    gdf = gpd.read_file(file)
    out_gdf = gdf.to_crs(crs=crs.to_string(), epsg=epsg, inplace=False)
    return out_gdf


# endregion: reproject vector data
