import numpy as np
from typing import Literal, Dict, Tuple, Union, Optional

from mpm_input_preprocessing.common.utils_helper import (
    _cast_array_to_minimum_dtype,
    _update_nodata
)


def __impute_min(
    src_array: np.ndarray,
) -> np.ndarray:
    impute_value = np.nanmin(src_array)
    return np.where(np.isnan(src_array), impute_value, src_array)


def __impute_max(
    src_array: np.ndarray,
) -> np.ndarray:
    impute_value = np.nanmax(src_array)
    return np.where(np.isnan(src_array), impute_value, src_array)


def __impute_mean(
    src_array: np.ndarray,
) -> np.ndarray:
    impute_value = np.nanmean(src_array)
    return np.where(np.isnan(src_array), impute_value, src_array)


def __impute_median(
    src_array: np.ndarray,
) -> np.ndarray:
    impute_value = np.nanmedian(src_array)
    return np.where(np.isnan(src_array), impute_value, src_array)


def __impute_zero(
    src_array: np.ndarray,
) -> np.ndarray:
    impute_value = 0
    return np.where(np.isnan(src_array), impute_value, src_array)


def __impute_custom(
    src_array: np.ndarray,
    custom_value: Union[int, float],
) -> np.ndarray:
    impute_value = custom_value
    return np.where(np.isnan(src_array), impute_value, src_array)


def impute(
    src: Tuple[np.ndarray, Dict],
    template: Optional[Tuple[np.ndarray, Dict]],
    method: Literal["min", "max", "mean", "median", "zero", "custom"],
    custom_value: Optional[int | float] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Replace nodata values within a 2D-array with a pre-defined method or custom value.

    Args:
        src: A tuple containing the source array and its metadata.
            - src_array: The source array to be transformed.
            - src_meta: Metadata associated with the source array, including 'nodata' value.
        method: The transformation method to apply.
            - "min": Replace missing values with array's minimum.
            - "max": Replace missing values with array's maximum.
            - "mean": Replace missing values with array's mean.
            - "median": Replace missing values with array's median.
            - "zero": Replace missing values with zero values.
            - "custom":  Replace missing values with user-input
        template: A tuple containing the template raster array and its metadata.
            - np.ndarray: The template raster data array.
            - Dict: The template raster metadata.
        custom_value: The value to replace missing values with when using the "custom" method.

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

    # Impute
    if method == "min":
        out_array = __impute_min(src_array)
    elif method == "max":
        out_array = __impute_max(src_array)
    elif method == "mean":
        out_array = __impute_mean(src_array)
    elif method == "median":
        out_array = __impute_median(src_array)
    elif method == "zero":
        out_array = __impute_zero(src_array)
    elif method == "custom":
        assert custom_value is not None, "Custom value must be provided for custom imputation method."
        out_array = __impute_custom(src_array, custom_value)
    else:
        raise ValueError(f"Invalid imputation method: {method}")

    # Update nodata
    out_nodata = _update_nodata(out_array, src_nodata)

    # Mask with template
    if template is not None:
        template_array, template_meta = template
        out_array = np.where(template_array == template_meta["nodata"], out_nodata, out_array)

    # Cast to minimum dtype and update metadata
    out_array, out_nodata = _cast_array_to_minimum_dtype(out_array, out_nodata)

    out_meta = src_meta.copy()
    out_meta.update(
        nodata=out_nodata,
        dtype=out_array.dtype
    )

    return out_array, out_meta
