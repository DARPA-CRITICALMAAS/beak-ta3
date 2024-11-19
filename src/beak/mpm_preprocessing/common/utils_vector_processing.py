import geopandas as gpd
import numpy as np
from typing import Tuple, Optional, List, Union
import rasterio
from rasterio import features, transform
from rasterio.enums import MergeAlg

from mpm_input_preprocessing.common.utils_helper import (
    _cast_array_to_minimum_dtype,
    _initialize_data_for_rasterization
)


def prepare_geodataframe(
    src: gpd.GeoDataFrame,
    template: Tuple[np.ndarray, dict],
    query: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Prepare a GeoDataFrame for rasterization.

    Ensure that the input GeoDataFrame is not empty, optionally filters it based on a query,
    and reproject's it to the coordinate reference system (CRS) of the provided template raster.

    Args:
        src: The input GeoDataFrame to be prepared for rasterization.
        template: A tuple containing the template array and its metadata.
            - template_array: The array representing the template raster.
            - template_meta: The metadata dictionary for the template raster.
        query: An optional query string to filter the GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: The queried and reprojected GeoDataFrame

    Raises:
        AssertionError: If the input GeoDataFrame is empty.
        AssertionError: If the queried GeoDataFrame is empty.
    """
    _, template_meta = template
    assert not src.empty, "Input GeoDataFrame is empty."

    if query is not None:
        src = src.query(query)
        assert not src.empty, "Query returned empty GeoDataFrame."

    return src.to_crs(crs=template_meta["crs"], inplace=False)


def rasterize_binary(
    src: gpd.GeoDataFrame,
    template: Tuple[np.ndarray, dict],
    custom_fill_value: Optional[int | float] = None,
    custom_nodata_value: Optional[int | float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Convert features into a binary raster.

    Use cases:
        - Template (coverage) creation: use custom_fill_value of None.
        - Encoding categorical features: use custom_fill_value of 0.
        - Label creation: use custom_fill_value of 0.

    Args:
        src: The input GeoDataFrame containing geometries to rasterize.
        template: A tuple containing the template array and its metadata.
        custom_fill_value: Custom fill value for the raster.
        custom_nodata_value: Custom nodata value for the raster.
            If not provided, the nodata value will be set to the minimum of the data type.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing the rasterized array and its metadata.

    Raises:
        AssertionError: If CRS mismatch between source and template.
    """
    _, template_meta = template
    assert src.crs == template_meta["crs"], "CRS mismatch between source and template."

    geometries = src.geometry.values
    values = np.ones_like(geometries, dtype=np.int8)

    out_array, out_meta = _rasterize_features(
        src=(geometries, values),
        template=template,
        custom_fill_value=custom_fill_value,
        custom_nodata_value=custom_nodata_value,
    )

    return out_array, out_meta


def rasterize_encode_categorical(
    src: gpd.GeoDataFrame,
    template: Tuple[np.ndarray, dict],
    column_to_raster: str,
    custom_nodata_value: Optional[int | float] = None,
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Rasterize a categorical column from a GeoDataFrame into multiple binary rasters.

    Takes a GeoDataFrame and a specified column containing categorical data,
    and rasterizes each unique value in that column into a separate binary raster indicating
    presence (1) or absence (0) of that value in the original geometries.

    Args:
        src: The input GeoDataFrame containing geometries and categorical data.
        template: A tuple containing the template array and its metadata.
        column_to_raster: The name of the column in the GeoDataFrame to rasterize.
        custom_nodata_value: Custom nodata value for the raster.
            If not provided, the nodata value will be set to the minimum of the data type.

    Returns:
        out_encodings: A list of tuples, each containing:
            - The name of the raster (str) in the format "column_name_value".
            - The rasterized array (np.ndarray).
            - The metadata dictionary (dict) for the raster.
    """
    out_encodings = []
    unique_values = src[column_to_raster].dropna().unique()

    for value in unique_values:
        geometries = src[src[column_to_raster] == value].geometry
        geometries = gpd.GeoDataFrame(geometries, crs=src.crs)

        out_name = column_to_raster + "_" + str(value)
        out_raster, out_meta = rasterize_binary(
            src=geometries,
            template=template,
            custom_fill_value=0,
            custom_nodata_value=custom_nodata_value,
        )

        result = (out_name, out_raster, out_meta)
        out_encodings.append(result)

    return out_encodings


def rasterize_continuous(
    src: gpd.GeoDataFrame,
    template: Tuple[np.ndarray, dict],
    column_to_raster: str,
    custom_nodata_value: Optional[int | float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Rasterize continuous values from a specified column in a GeoDataFrame.

    Takes a GeoDataFrame and a specified column containing continuous data,
    and rasterizes the values into a raster array based on a given template.

    Args:
        src: The input GeoDataFrame containing geometries and continuous data.
        template: A tuple containing the template raster array and its metadata.
            - np.ndarray: The template raster data array.
            - dict: The template raster metadata.
        column_to_raster: The name of the column in the GeoDataFrame to rasterize.
        custom_nodata_value: Custom nodata value for the raster.
            If not provided, the nodata value will be set to the minimum of the data type.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing:
            - np.ndarray: The rasterized array.
            - dict: The metadata dictionary for the rasterized array.
    """
    geometries = src.geometry
    values = src[column_to_raster].values
    _, template_meta = template

    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.floating)

    values_min_dtype = rasterio.dtypes.get_minimum_dtype(values)
    values = values.astype(values_min_dtype)

    out_raster, out_meta = _rasterize_features(
        src=(geometries, values),
        template=template,
        custom_nodata_value=custom_nodata_value,
    )

    return out_raster, out_meta


def _get_geometry_type(src: gpd.GeoDataFrame) -> str:
    """
    Determine the geometry type of a GeoDataFrame.

    Checks the geometry type of all geometries in the provided GeoDataFrame and returns a string
    indicating the type of geometry. It supports 'Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon',
    and 'MultiPolygon'.

    Args:
        src: The GeoDataFrame whose geometry type is to be determined.

    Returns:
        geometry_type: A string indicating the geometry type of the GeoDataFrame.

    Raises:
        ValueError: If the GeoDataFrame contains mixed or unsupported geometry types.
    """
    if src.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        geometry_type = "point"
    elif src.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        geometry_type = "line"
    elif src.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        geometry_type = "polygon"
    else:
        raise ValueError("The GeoDataFrame contains mixed or unsupported geometry types.")

    return geometry_type


def _transform_from_geometries(
    src: gpd.GeoDataFrame,
    resolution: Union[int, float]
) -> Tuple[int, int, transform.Affine]:
    """
    Calculate the transform parameters required to convert the input geometries to a specified resolution.

    Args:
        src: The input GeoDataFrame containing the geometries.
        resolution: The desired resolution for the transformation.

    Returns:
        Tuple[float, float, transform.Affine]: A tuple containing the width, height, and transform
            parameters required for the transformation.

    """
    min_x, min_y, max_x, max_y = src.total_bounds

    out_width = int((max_x - min_x) / resolution)
    out_height = int((max_y - min_y) / resolution)

    out_transform = transform.from_origin(min_x, max_y, resolution, resolution)
    return out_width, out_height, out_transform


def _rasterize_features(
    src: Tuple[gpd.array.GeometryArray, np.ndarray],
    template: Tuple[np.ndarray, dict],
    custom_fill_value: Optional[int | float] = None,
    custom_nodata_value: Optional[int | float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Rasterize geometries into a raster array based on a given template.

    Args:
        src: A tuple containing geometries and values.
            - GeometryArray: An array of geometries to be rasterized.
            - np.ndarray: An array of values corresponding to the geometries.
        template: A tuple containing the template raster array and its metadata.
            - np.ndarray: The template raster data array.
            - Dict: The template raster metadata.
        custom_fill_value: A custom fill value for pixels without geometries.
        custom_nodata_value: A custom nodata value for the raster.
            If not provided, the nodata value will be set to the minimum of the data type.

    Returns:
        A tuple containing:
            - out_array: The rasterized array.
            - out_meta: The metadata dictionary for the rasterized array.
    """
    src_geometries, src_values = src
    template_array, template_meta = template

    # Init nodata and fill values
    src_values, nodata_value, fill_value = _initialize_data_for_rasterization(
        src_values,
        custom_nodata_value,
        custom_fill_value
    )

    # Rasterize and update metadata
    raster_array = features.rasterize(
        shapes=list(zip(src_geometries, src_values)),
        out_shape=(template_meta["height"], template_meta["width"]),
        fill=fill_value,
        transform=template_meta["transform"],
        all_touched=False,
        merge_alg=getattr(MergeAlg, "replace"),
        default_value=1,
    )

    # Mask array with template nodata, update metadata
    out_array = np.where(template_array == template_meta["nodata"], nodata_value, raster_array)
    out_array, out_nodata = _cast_array_to_minimum_dtype(out_array, nodata_value, unify_integer_types=True)

    out_meta = template_meta.copy()
    out_meta.update(
        nodata=out_nodata,
        dtype=out_array.dtype,
        count=1
    )

    return out_array, out_meta
