import numpy as np
from typing import Literal, Dict, Tuple

from mpm_input_preprocessing.common.utils_helper import (
    _cast_array_to_minimum_dtype,
    _update_nodata
)


def __transform_log(
    src_array: np.ndarray,
) -> np.ndarray:
    src_array = np.where(src_array > 0, src_array, np.nan)
    return np.log(src_array)


def __transform_log1p(
    src_array: np.ndarray,
) -> np.ndarray:
    src_array = np.where(src_array < 0, np.nan, src_array)
    return np.log1p(src_array)


def __transform_abs(
    src_array: np.ndarray,
) -> np.ndarray:
    return np.abs(src_array)


def __transform_sqrt(
    src_array: np.ndarray,
) -> np.ndarray:
    src_array = np.where(src_array < 0, np.nan, src_array)
    return np.sqrt(src_array)


def __transform_minmax(
    src_array: np.ndarray,
) -> np.ndarray:
    array_min = np.nanmin(src_array)
    array_max = np.nanmax(src_array)

    # Pass source array to zero if min and max are the same, else transform
    if array_min == array_max:
        return src_array
    else:
        return (src_array - array_min) / (array_max - array_min)


def __transform_maxabs(
    src_array: np.ndarray,
) -> np.ndarray:
    array_absmax = np.nanmax(
        np.abs(src_array)
    )
    return src_array / array_absmax


def __transform_std(
    src_array: np.ndarray,
) -> np.ndarray:
    array_mean = np.nanmean(src_array)
    array_sd = np.nanstd(src_array)

    # Pass source array to zero if standard deviation is zero, else transform
    if array_sd == 0:
        return src_array
    else:
        return (src_array - array_mean) / array_sd


def transform(
    src: Tuple[np.ndarray, Dict],
    method: Literal["log", "log1p", "abs", "sqrt", "minmax", "maxabs", "standard"]
) -> Tuple[np.ndarray, Dict]:
    """
    Apply a specified mathematical transformation to a source array.

    Sets new nodata if the provided value is within the transformed array's range.

    Args:
        src: A tuple containing the source array and its metadata.
            - src_array: The source array to be transformed.
            - src_meta: Metadata associated with the source array, including 'nodata' value.
        method: The transformation method to apply.
            - "log": Apply natural logarithm to the array.
            - "log1p": Apply natural logarithm to (array + 1).
            - "abs": Apply absolute value to the array.
            - "sqrt": Apply square root to the array.
            - "minmax": Normalize the array to the range [0, 1].
            - "maxabs": Normalize the array to maximum bounds [-1, 1].
            - "standard": Normalize the array to have zero mean and unit variance.

    Returns:
        A tuple containing the transformed array and its updated metadata.
            - out_array: The transformed array.
            - out_meta: Updated metadata for the transformed array.

    Raises:
        ValueError: If the specified transformation method is not supported.
    """
    src_array, src_meta = src
    src_nodata = src_meta["nodata"]
    src_array = np.where(src_array == src_nodata, np.nan, src_array)

    if method == "log":
        out_array = __transform_log(src_array)
    elif method == "log1p":
        out_array = __transform_log1p(src_array)
    elif method == "abs":
        out_array = __transform_abs(src_array)
    elif method == "sqrt":
        out_array = __transform_sqrt(src_array)
    elif method == "minmax":
        out_array = __transform_minmax(src_array)
    elif method == "maxabs":
        out_array = __transform_maxabs(src_array)
    elif method == "standard":
        out_array = __transform_std(src_array)
    else:
        raise ValueError(f"Invalid transform method: {method}")

    out_nodata = _update_nodata(out_array, src_nodata)
    out_array = np.nan_to_num(out_array, nan=out_nodata)
    out_array, out_nodata = _cast_array_to_minimum_dtype(out_array, out_nodata)

    out_meta = src_meta.copy()
    out_meta.update(
        nodata=out_nodata,
        dtype=out_array.dtype
    )

    return out_array, out_meta
