import numpy as np
from typing import Tuple, Union, Dict, Literal

from mpm_input_preprocessing.common.utils_helper import (
    _cast_array_to_minimum_dtype,
    _update_nodata
)

def __get_outliers_iqr(
    src_array: np.ndarray,
    threshold: Union[int, float],
) -> Tuple[int | float, int | float]:
    """
    Get outliers based on the IQR method.

    Args:
        src_array: The source array for outlier detection.
        threshold: The threshold value for outlier detection.

    Returns:
        Tuple[lower_bound, upper_bound,np.ndarray: The detected outliers.
    """
    # Calculate quantiles
    Q1 = np.nanpercentile(src_array, 25)
    Q3 = np.nanpercentile(src_array, 75)

    # IQR
    IQR = Q3 - Q1

    # Lower and upper bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    return lower_bound, upper_bound


def clip_outliers(
    src: Tuple[np.ndarray, Dict],
    method: Literal["iqr"]
) -> Tuple[np.ndarray, Dict]:
    """
    Clip outliers in the provided array.

    Currently, supports only the IQR method, but designed for adding more methods if necessary.

    Args:
        src: A tuple containing the source array and its metadata.
            - src_array: The source array to be transformed.
            - src_meta: Metadata associated with the source array, including 'nodata' value.
        method: The method for outlier detection.
            - "iqr": Use the Interquartile Range (IQR) method for outlier detection with a threshold of 1.5.

    Returns:
        A tuple containing the outlier-corrected array and its updated metadata.
            - out_array: The transformed array.
            - out_meta: Updated metadata for the transformed array.
    Raises:
        ValueError: If the specified method is not supported.
    """
    src_array, src_meta = src
    src_nodata = src_meta["nodata"]
    src_array = np.where(src_array == src_nodata, np.nan, src_array)

    if method == "iqr":
        lower_bound, upper_bound = __get_outliers_iqr(src_array, threshold=1.5)
    else:
        raise ValueError(f"Invalid transform method: {method}")

    out_array = np.clip(src_array, a_min=lower_bound, a_max=upper_bound)
    out_nodata = _update_nodata(out_array, src_nodata)
    out_array = np.nan_to_num(out_array, nan=out_nodata)
    out_array, out_nodata = _cast_array_to_minimum_dtype(out_array, out_nodata)

    out_meta = src_meta.copy()
    out_meta.update(
        nodata=out_nodata,
        dtype=out_array.dtype
    )

    return out_array, out_meta
