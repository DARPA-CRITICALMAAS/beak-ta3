import urllib.request
import io
import ssl

import rasterio
import numpy as np
from typing import Dict, Tuple, Union, Optional


def download_file_from_cdr(download_url: str) -> io.BytesIO:
    """
    Download a raster file from the specified URL.

    This function attempts to download a raster file from the given URL.
    If the download fails due to SSL verification issues, it retries the download without SSL verification.

    Args:
        download_url: The URL from which to download the raster file.

    Returns:
        io.BytesIO: A BytesIO object containing that can be used as path-like input.
    """
    try:
        # With SSL
        response = urllib.request.urlopen(download_url)
    except:
        # Remove SSL verification if try fails
        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(download_url, context=context)

    return io.BytesIO(response.read())


def read_raster_band(raster: rasterio.io.DatasetReader, band: int = 1, **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Read a raster file from the given DatasetReader object and returns the raster array and its metadata.

    In preparation for iterating over multiple bands, the kwargs argument can take every valid key-value pair
    that is present in the input raster's metadata dictionary. E.g., if different bands have different
    nodata values or data types, the metadata will be updated for further processing.

    Args:
        raster: The DatasetReader object from which to read the raster.
        band: The band number to read from the raster.
        **kwargs: Additional keyword arguments to update the metadata of the raster.

    Returns:
        Tuple of np.ndarray, Dict: The raster array and its updated metadata.
            - np.ndarray: A 2-dimensional raster array read from the input raster.
            - Dict: The (updated) metadata of the raster read from the input raster.
    """
    out_array = raster.read(band)
    out_meta = raster.meta.copy()

    out_meta.update(
        count=1
    )
    out_meta.update(kwargs)

    return out_array, out_meta


def save_raster(
    src_array: np.ndarray,
    src_meta: Dict,
    output_path: str,
    compatibility_mode: bool = False,
    **kwargs
):
    """
    Save a raster array to a file with the specified metadata or profile.

    Writes a given raster array to a file at the specified output path, using the provided metadata.
    It ensures that the minimum data type for the raster array is set and updates the metadata with any additional
    keyword arguments provided. The metadata can be replaced by the raster's profile, providing additional parameters
    for tiling and block size in order to optimize storage and processing for COG compatibility.

    Nodata value in the kwargs argument will overwrite the automatically determined nodata value.

    Args:
        src_array: The raster array to be saved.
        src_meta: The metadata dictionary for the raster.
        output_path: The file path where the raster will be saved.
        compatibility_mode: A boolean indicating whether to enable QGIS compatibility for 8bit signed integers.
            the profile will be updated to support Cloud Optimized GeoTIFF (COG).
        **kwargs: Additional keyword arguments to update the metadata of the raster.

    Returns:
        None
    """
    # Prepare data
    raster_array = np.expand_dims(src_array, axis=0) if src_array.ndim == 2 else src_array
    raster_meta = src_meta.copy()
    raster_meta.update(kwargs)

    # Ensure data types and values
    raster_array_dtype = raster_array.dtype.name
    raster_array_dtype_min, raster_array_dtype_max = __get_dtype_range_info(raster_array_dtype)
    raster_array_min, raster_array_max = np.min(raster_array), np.max(raster_array)
    meta_nodata = raster_meta["nodata"]

    # Assertions
    assert raster_array_dtype_min <= raster_array_min <= raster_array_dtype_max, "Array minimum out of range."
    assert raster_array_dtype_min <= raster_array_max <= raster_array_dtype_max, "Array maximum out of range."
    assert raster_array_dtype_min <= meta_nodata <= raster_array_dtype_max, "Nodata value out of range."

    if compatibility_mode is True:
        raster_array_dtype = "int16" if raster_array_dtype == "int8" else raster_array_dtype
        raster_array = raster_array.astype(raster_array_dtype)

        dtype_min, _ = __get_dtype_range_info(raster_array_dtype)
        raster_array = np.where(raster_array == meta_nodata, dtype_min, raster_array)

        raster_meta.update(
            nodata=dtype_min,
            dtype=raster_array_dtype
        )

    # Create file
    with rasterio.open(output_path, "w", **raster_meta) as dst:
        for i in range(0, raster_meta["count"]):
            dst.write(raster_array[i], i + 1)
        dst.close()


def _update_nodata(
    array: np.ndarray,
    nodata_value: Union[int, float],
):
    """
    Update nodata value based on the array's values.

    Sets the nodata value to a boundary value of the array if the provided nodata value is
    within the array's range. Does not update anything else than this value.

    Args:
        array: The input array to be adjusted.
        nodata_value: The nodata value.

    Returns:
        The updated nodata value.
    """
    out_dtype = __get_minimum_dtype(array)
    edge_value = __get_dtype_range_value(out_dtype)
    values_min, values_max = np.nanmin(array), np.nanmax(array)

    if nodata_value == values_min or nodata_value == values_max:
        values_dtype = __get_next_higher_dtype(array.dtype)
        out_nodata = __get_dtype_range_value(values_dtype)
    else:
        out_nodata = edge_value

    return out_nodata


def _initialize_data_for_rasterization(
    array: np.ndarray,
    nodata_value: Optional[int | float],
    fill_value: Optional[int | float],
) -> Tuple[np.ndarray, Union[int, float], Union[int, float]]:
    """
    Initialize data for rasterization by adjusting the array data type and handling nodata and fill values.

    If the nodata value is None, nodata value will be initialized by the array's and fill values.
    The array will be casted to the minimum possible dtype and nodata will be updated to an edge value:
        - for unsigned integers: positive boundary
        - for signed integers: negative boundary
        - for floating point numbers: negative boundary

    If the nodata values is not None, the array will be casted to the minimum possible type,
    without changing the nodata value.

    Args:
        array: Input data array.
        nodata_value: Value representing empty pixels.
        fill_value: Value used to fill areas.

    Returns:
        Tuple[np.ndarray, Union[int, float], Union[int, float]]: Tuple containing
            the processed array, nodata value, and fill value.
    """

    # Ensure array has the minimum required dtype
    array = array.astype(__get_minimum_dtype(array))

    # Initialize nodata and fill values
    nodata = _cast_value_to_int(nodata_value) if nodata_value is not None else None
    fill = _cast_value_to_int(fill_value) if fill_value is not None else None

    # Prepare values for dtype determination
    values = [
        np.min(array), np.max(array)
    ]

    if fill is not None:
        values.append(fill)

    # Determine minimum dtype for values and its edge value
    values = np.array(values)
    values_min, values_max = np.min(values), np.max(values)
    values_dtype = __get_minimum_dtype(values)
    dtype_edge = __get_dtype_range_value(values_dtype)

    # Set default nodata to edge if not provided
    if nodata is None:
        nodata = dtype_edge

        if nodata == values_min or nodata == values_max:
            values_dtype = __get_next_higher_dtype(values_dtype)

        nodata = __get_dtype_range_value(values_dtype)
        array = array.astype(values_dtype)
    else:
        array, _ = _cast_array_to_minimum_dtype(array, nodata)

    # Set fill to nodata if not provided
    if fill is None:
        fill = nodata

    return array, nodata, fill


def _cast_value_to_int(value: Union[int, float]) -> Union[int, float]:
    """
    Cast a value to integer type if possible.

    Maximum to cast are "uint64" and "int64" types, which both return True when passed through np.can_cast().

    Args:
        value: The value to cast to integer type.

    Returns:
        Union[int, float]: The value cast to integer type if possible, otherwise the original value.
    """
    max_dtypes = ["uint64", "int64"]

    dtype_range_list = []
    for dtype in max_dtypes:
        dtype_range = __get_dtype_range_info(dtype)
        dtype_range_list.append(dtype_range)

    all_mins = [min(range_pair) for range_pair in dtype_range_list]
    all_maxs = [max(range_pair) for range_pair in dtype_range_list]

    overall_min = min(all_mins)
    overall_max = max(all_maxs)

    if type(value) is float and value.is_integer() and (overall_min <= value <= overall_max):
        out_value = int(value)
    else:
        out_value = value

    return out_value


def _cast_array_to_minimum_dtype(
    array: np.ndarray,
    value: Optional[int | float] = None,
    unify_integer_types: bool = False,
) -> Tuple[np.ndarray, Optional[int | float]]:
    """
    Determine the minimum data type for the raster array.

    Args:
        array: The raster array to determine the minimum data type for.
        value: The value for additional comparison.

    Returns:
        Tuple of np.ndarray, Dict: The raster array and its metadata.
            - np.ndarray: The raster array with the minimum data type determined.
            - Dict: The updated metadata dictionary with the minimum data type set.
    """
    # Init
    cast_dtype = __get_minimum_dtype(array, unify_integer_types)

    # New minimum data type
    cast_dtype_min, cast_dtype_max = __get_dtype_range_info(cast_dtype)

    # Ensure consistency with nodata value
    if not cast_dtype_min <= value <= cast_dtype_max:
        cast_dtype = array.dtype

    if value is not None:
        # Cast value to integer type if possible
        value = _cast_value_to_int(value)

        # Promote dtype to match nodata value
        value_dtype = np.min_scalar_type(value)
        promoted_dtype = np.promote_types(cast_dtype, value_dtype)

        # Ensure compatibility with rasterio dtypes
        cast_dtype = __get_minimum_dtype(
            array.astype(promoted_dtype),
            unify_integer_types
        )

        if np.issubdtype(cast_dtype, np.floating):
            value = float(value)
        else:
            value = int(value)

    return array.astype(cast_dtype), value


def __get_dtype_range_info(
    dtype: Union[str, np.dtype],
) -> Tuple[np.number, np.number]:
    """
    Determine the range of values for the raster array.

    Replaces the rasterio.dtypes.get_dtype_range_info() since this function returns "float64"
    type for min and max values of "float32".

    Args:
        dtype: A dtype to determine the range for.

    Returns:
        Tuple of np.number, np.number: The minimum and maximum values to the related data type.

    Raises:
        ValueError: If the dtype is not supported.
    """
    if np.issubdtype(dtype, np.integer):
        dtype_min = np.iinfo(dtype).min
        dtype_max = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        dtype_min = np.finfo(dtype).min
        dtype_max = np.finfo(dtype).max
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    return dtype_min, dtype_max


def __get_dtype_range_value(dtype: Union[str, np.dtype]) -> Union[int, float]:
    """
    Get the range value by dtype.

    Args:
        dtype: The dtype to get the range value for.

    Returns:
        Union[int, float]: The range value for the given dtype.
    """
    if np.issubdtype(dtype, np.unsignedinteger):
        dtype_info = np.iinfo(dtype)
        out_value = dtype_info.max
    else:
        dtype_info = np.iinfo(dtype) if np.issubdtype(dtype, np.signedinteger) else np.finfo(dtype)
        out_value = dtype_info.min

    return out_value


def __get_next_higher_dtype(dtype: Union[str, np.dtype]) -> np.dtype:
    """
    Get the next higher dtype.

    Args:
        dtype: The dtype to get the next higher one for.

    Returns:
        np.dtype: The next higher dtype. If input dtype is not supported, return the original dtype.
    """
    dtype = dtype.name if not isinstance(dtype, str) else dtype

    if dtype == "uint1":
        return np.uint8
    elif dtype == "uint4":
        return np.uint8
    elif dtype == "uint8":
        return np.uint16
    elif dtype == "int8":
        return np.int16
    elif dtype == "uint16":
        return np.uint32
    elif dtype == "int16":
        return np.int32
    elif dtype == "uint32":
        return np.uint64
    elif dtype == "int32":
        return np.int64
    elif dtype == "float32":
        return np.float64
    else:
        return np.dtype(dtype)


def __get_minimum_dtype(
    array: np.ndarray,
    unify_integer_types: bool = False,
) -> str:
    """
    Determine the minimum data type for the raster array.

    Comparison based on the raster's minimum and maximum values.
    Alternative to rasterio.dtypes._get_minimum_dtype() since it does not return "int8".
    According to the restrictions, smallest bit-size for outputs is
        - 8 for integer
        - 32 for floats

    Restrictions:
        - dtypes smaller than "uint8" are not supported by numpy and rasterio
        - dtype "float16" is not yet supported rasterio

    Args:
        array: The raster array to determine the minimum data type for.
        unify_integer_types: Whether to differentiate signed and unsigned integers.

    Returns:
        str: The minimum data type for the raster array.

    Raises:
        ValueError: If the dtype is not supported.
    """
    signedinteger_dtypes = ["int8", "int16", "int32",  "int64"]
    unsignedinteger_dtypes = ["uint8", "uint16","uint32", "uint64"]
    integer_dtypes = [dtype for pair in zip(unsignedinteger_dtypes, signedinteger_dtypes) for dtype in pair]

    floating_dtypes = [
        "float32", "float64"
    ]

    # Init
    src_min, src_max = np.nanmin(array), np.nanmax(array)

    # Integer
    if unify_integer_types is True:
        if np.issubdtype(array.dtype, np.integer):
            assert src_min >= np.iinfo("int64").min and src_max <= np.iinfo("uint64").max, \
                "Values out of range for supported integer type."

            return next(
                dtype for dtype in integer_dtypes if (
                    np.iinfo(dtype).min <= src_min and np.iinfo(dtype).max >= src_max
                )
            )
    else:
        if np.issubdtype(array.dtype, np.signedinteger):
            assert src_min >= np.iinfo("int64").min and src_max <= np.iinfo("int64").max, \
                "Values out of range for supported signed integers type."

            return next(
                dtype for dtype in signedinteger_dtypes if (
                    np.iinfo(dtype).min <= src_min and np.iinfo(dtype).max >= src_max
                )
            )

        if np.issubdtype(array.dtype, np.unsignedinteger):
            assert src_min >= np.iinfo("uint64").min and src_max <= np.iinfo("uint64").max, \
                "Values out of range for supported unsigned integers type."

            return next(
                dtype for dtype in unsignedinteger_dtypes if (
                    np.iinfo(dtype).min <= src_min and np.iinfo(dtype).max >= src_max
                )
            )

    # Floating
    if np.issubdtype(array.dtype, np.floating):
        assert src_min >= np.finfo("float64").min and src_max <= np.finfo("float64").max, \
            "Values out of range for supported floating point type."

        return next(
            dtype for dtype in floating_dtypes if np.finfo(dtype).min <= src_min and np.finfo(dtype).max >= src_max
        )

    # Else case
    raise ValueError(f"Unsupported data type: {array.dtype}")
