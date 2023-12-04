import numpy as np
import pandas as pd
import scipy
from typing import List, Tuple


# References
# Some non-trivial functionalities were adapted from other sources.
# The original sources are listed below and referenced in the code as well.
#
# EIS toolkit:
# GitHub repository https://github.com/GispoCoding/eis_toolkit under EUPL-1.2 license.


def fill_nodata_with_mean(
    array: np.ndarray,
    nodata_value: np.number,
    size: int = 3,
    num_nan_max: int = 4,
) -> np.ndarray:
    """
    Fill nodata values with the mean from surrounding cells.

    Args:
        array (np.ndarray): Input array with nodata values.
        nodata_value (np.number): Value representing nodata in the array.
        size (int): Size of the kernel used for calculating the mean. Defaults to 3.
        num_nan_max (int): Maximum number of nodata cells allowed in the kernel neighborhood for mean calculation. Defaults to 4.

    Returns:
        np.ndarray: Array with nodata values imputed by the mean from surrounding cells.
    """
    # Set kernel size
    kernel = np.ones((size, size))

    # Create mask for nodata values and convert to int
    nan_mask = np.isin(array, nodata_value)

    # Create array for sum of np.nan values in kernel neighborhood
    nan_sum = scipy.ndimage.generic_filter(
        nan_mask.astype(int), np.sum, footprint=kernel, mode="constant", cval=0
    )

    # Create combined masked with certain amount of nodata cells allowed for mean calculation
    nan_sum_mask = np.logical_and(nan_mask, nan_sum <= num_nan_max)

    # Initialize output array
    out_array = np.where(nan_mask, np.nan, array)

    # Calculate mean for each cell in kernel neighborhood based on nan_sum_mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        out_array = np.where(
            nan_sum_mask,
            scipy.ndimage.generic_filter(
                out_array, np.nanmean, footprint=kernel, mode="reflect"
            ),
            out_array,
        )

    return np.where(np.isnan(out_array), nodata_value, out_array)

def create_encodings_from_dataframe(
    value_columns: List[str],
    data: pd.DataFrame,
    export_absent: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create encodings for categorical data.

    Args:
        value_columns (List[str]): List of column names containing categorical values.
        data (pd.DataFrame): Input data frame.
        export_absent (bool): Flag indicating whether to export "Absent" columns.

    Returns:
        pd.DataFrame: Encoded data frame.
        List[str]: List of new column names.

    """
    # Create binary encodings
    data_encoded = pd.get_dummies(data[value_columns], prefix=value_columns).astype(
        np.uint8
    )

    # Get new column names
    new_value_columns = data_encoded.columns.to_list()

    if export_absent == False:
        # Get rid of the "Absent" columns since they are not needed in binary encoding
        new_value_columns = [
            column for column in new_value_columns if "Absent" not in column
        ]
        data_encoded = data_encoded[new_value_columns]

    return data_encoded, new_value_columns
