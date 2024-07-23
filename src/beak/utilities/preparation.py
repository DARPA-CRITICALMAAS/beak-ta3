import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

from pathlib import Path
from sklearn.impute import SimpleImputer
from beartype.typing import List, Tuple, Union, Optional, Literal, Sequence
from numbers import Number

from beak.utilities.io import load_raster, read_raster
from beak.utilities.focal import _create_local_buffer_binary_version, _create_local_buffer_generic_version


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


def impute_data(
    data: Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[str]] = None,
    strategy: str = "mean",
    fill_value: Union[Number, str] = None,
    missing_values: Optional[Number] = np.nan,
) -> Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame]:
    """
    Imputes missing values in the given data using the specified strategy.

    For arrays, the data needs to be in the shape (rows, columns), e.g. 3 layers with 5 entries each: (5, 3).

    Args:
        data (Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame]): The data to be imputed.
        columns (List[str]): The columns to be imputed.
        strategy (str, optional): The imputation strategy. Defaults to "mean".
        fill_value (Union[Number, str], optional): The value to fill missing values with. Defaults to None.
        missing_values (Optional[Number], optional): The value to be treated as missing. Defaults to np.nan.

    Returns:
        Union[np.ndarray, pd.DataFrame, gpd.GeoDataFrame]: The imputed data.
    """
    imputer = SimpleImputer(
        strategy=strategy, missing_values=missing_values, fill_value=fill_value
    )
    data_imputed = data

    if isinstance(data, np.ndarray):
        nan_count = np.sum(np.isnan(data_imputed))
        if nan_count > 0:
            data_shape = data_imputed.shape
            data_imputed = data_imputed.reshape(-1, 1)
            data_imputed = imputer.fit_transform(data_imputed)
            data_imputed = data_imputed.reshape(data_shape)
        else:
            data_imputed = data
    elif isinstance(data, pd.DataFrame) or isinstance(data, gpd.GeoDataFrame):
        data_imputed[columns] = imputer.fit_transform(data_imputed[columns])
    return data_imputed


def delete_nan_elements(array: np.ndarray, transpose: bool = True) -> np.ndarray:
    """
    Deletes NaN elements from the given array.

    Args:
        array (np.ndarray): The array to be processed.
        transpose (bool, optional): Flag indicating whether to transpose the array before processing. Defaults to True.

    Returns:
        np.ndarray: The processed array with NaN elements removed.
    """
    target_locations = np.isnan(array).any(axis=0)
    out_array = array[:, ~target_locations]

    if transpose is True:
        out_array = np.transpose(out_array)

    return out_array


def create_hard_buffer_around_labels(
    array: np.ndarray,
    radius: int = 1,
    shape: Literal["square", "circle"] = "circle",
    target_value: int = 1,
    buffer_value: Optional[Number] = None,
    overwrite_nodata: bool = False,
) -> np.ndarray:
    """
    Extends the positive values in the given array.

    A: If the buffer value is not specified, the buffer value will be the same as the positive labels value.
    B: If the buffer value is specified, elements around the positive labels will be replaced with this value.

    For extending positive labels, use option A.
    For excluding negative labels, use option B with values such as np.nan oder -1.

    Args:
        array (np.ndarray): The array to be extended.
        radius (int): The radius of the buffer (1 for a 3x3 window with 9 pixels, size is n*2 + 1).
            Defaults to 1.
        shape (Literal["square", "circle"]): The shape of the buffer.
            Defaults to "square".
        target_value (int): The value to be extended.
            Defaults to 1.
        buffer_value (Optional[Number]): The value to be used for the buffer.
            Defaults to None.
        overwrite_nodata (bool): Whether to extend the buffer into nodata cells.

    Returns:
        np.ndarray: The extended label's array.
    """
    out_array = np.copy(array)
    out_array = _create_local_buffer_binary_version(
        array=out_array,
        radius=radius,
        shape=shape,
        target_value=target_value,
    )

    buffer_value = target_value if buffer_value is None else buffer_value

    out_array = np.where(out_array == True, buffer_value, array)
    out_array = np.where(array == target_value, target_value, out_array)

    if overwrite_nodata is False:
        out_array = np.where(np.isnan(array), np.nan, out_array)

    return out_array


# region: Sampling functions
def _sampling_select_classes(
    array: np.ndarray,
    **kwargs,
):
    type = kwargs["type"] if "type" in kwargs.keys() else None
    threshold = kwargs["threshold"] if "threshold" in kwargs.keys() else None
    include = kwargs["include"] if "include" in kwargs.keys() else None
    exclude = kwargs["exclude"] if "exclude" in kwargs.keys() else None

    out_array = np.copy(array)
    if exclude is not None:
        exclude_mask = np.isin(out_array, exclude)
        out_array = np.where(exclude_mask, np.nan, out_array)

    if include is not None:
        include_mask = np.isin(out_array, include)
        out_array = np.where(include_mask, out_array, np.nan)

    if threshold is not None:
        unique_values = np.unique(out_array)
        include_classes = (
            np.nanquantile(unique_values, threshold, method="nearest")
            if threshold < 1
            else threshold
        )

        if type == "positives":
            out_array = np.where(array < include_classes, np.nan, out_array)
        elif type == "negatives":
            out_array = np.where(array < include_classes, out_array, np.nan)

    classes, counts = _get_unique_values_list(out_array)

    out_dict = {
        "type": type,
        "threshold": threshold,
        "classes": classes,
        "counts": counts,
        "counts_sum": np.sum(counts),
        "include": include,
        "exclude": exclude,
    }
    return out_array, out_dict


def _get_unique_values_list(
    array: np.ndarray,
    include_nan: bool = False,
    include_inf: bool = False,
    verbose=0,
) -> Tuple[Sequence[Number], Sequence[Number]]:
    """
    Get a list of unique values and their counts from a numpy array.

    Args:
        array (np.ndarray): The input array from which to extract unique values.
        include_nan (bool, optional): Whether to include NaN values in the unique values list. Defaults to False.
        verbose (int, optional): Whether to print the unique values and their counts. Defaults to 0 (no output).

    Returns:
        np.ndarray: A tuple containing two numpy arrays:
            - classes: The unique values in the array.
            - counts: The counts of each unique value in the array.
    """
    unique_values = list(np.unique(array, return_counts=True))

    if include_nan is False:
        no_nan_mask = ~np.isnan(unique_values[0])
        unique_values[0] = unique_values[0][no_nan_mask]

    if include_inf is False:
        no_inf_mask = ~np.isinf(unique_values[0])
        unique_values[0] = unique_values[0][no_inf_mask]

    classes = list(unique_values[0])
    counts = list(unique_values[1])

    while len(classes) < len(counts):
        counts = counts[:-1]

    if verbose == 1:
        for _class, _count in zip(classes, counts):
            print(f"Class: {_class}, Count: {_count}")
    elif verbose > 1:
        print("Verbose accepts only 0 or 1.")

    return classes, counts


def _sampling_select_random_points(
    array: np.ndarray,
    selection: Sequence[int],
    strategy: Union[Sequence[Number], Number],
    target_value: Optional[int] = None,
    merge_classes: bool = False,
    min_px: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Sequence[Number], Sequence[Number]]:
    """
    Select random points from a given array based on the specified strategy and method.

    Args:
        array (np.ndarray): The input array from which to select points.
        selection (Sequence[int]): The sequence of values to include in the class selection.
        strategy (Union[Sequence[Number], Number]): The sampling strategy, either a single value or a sequence of values.
            Values must be between 0 and 1, representing the fraction of pixels to select.
            If not a single value, values must be provided for each class in the same order as the classes.
        target_value (int): The value to assign to the selected points.
        merge_classes (bool): The method to use for sampling. Defaults to False.
            "True" will merge all pixels of the selected classes to one value before sampling.
            "False" will sample the pixels in each cluster separately.
        min_px (int, optional): The minimum number of pixels to select for each class. Defaults to 1.
        seed (Optional[int], optional): The seed for the random number generator. Defaults to None.

    Returns:
        Tuple[np.ndarray, dict]:
            - A numpy array with the selected points.
            - A dictionary with most important information about the sampling process and results.

    Raises:
        ValueError: If the strategy values are not between 0 and 1.
        ValueError: If the length of the strategy values does not match the number of classes.
        ValueError: If the provided array does not contain any valid values.
        ValueError: If the provided selection and the array classes do not match.
    """
    strategy = [strategy] if isinstance(strategy, Number) else strategy
    array_classes, array_counts = _get_unique_values_list(array)

    if min(strategy) < 0 or max(strategy) > 1:
        raise ValueError(
            f"The sampling strategy must always be between 0 and 1, but got {strategy}."
        )
    elif len(strategy) > 1 and len(strategy) != len(array_classes):
        raise ValueError(
            f"The length of the provide strategy values does not match the number of classes."
        )
    elif len(array_classes) == 0:
        raise ValueError(
            f"Provided array does not contain any valid values. Please check the provided array."
        )
    elif selection != array_classes:
        raise ValueError(
            f"Provided selection and the array classes do not match. Please check the provided classes or array."
        )

    sampling_array = array.copy().flatten()
    random_array = np.zeros_like(sampling_array)

    if merge_classes is True:
        merge_value = 1
        sampling_array = np.where(
            np.isin(sampling_array, selection), merge_value, np.nan
        )
        selection = [merge_value]
        array_counts = [sum(array_counts)]

    for idx, value in enumerate(selection):
        fraction_px = strategy[0] if len(strategy) == 1 else strategy[idx]
        number_px = int(np.round(fraction_px * array_counts[idx]))

        if number_px < min_px:
            number_px = min_px
        elif number_px > array_counts[idx]:
            number_px = array_counts[idx]

        class_idx = np.argwhere(sampling_array == value).flatten()
        random_idx = np.random.RandomState(seed).choice(
            class_idx, size=number_px, replace=False
        )

        random_array[random_idx] = 1

    sampling_classes, sampling_counts = _get_unique_values_list(
        np.where(random_array == 1, sampling_array, np.nan)
    )

    out_values = target_value if target_value is not None else array.flatten()
    out_array = np.where(random_array == 1, out_values, np.nan).reshape(array.shape)

    out_dict = {
        "classes": sampling_classes,
        "counts": sampling_counts,
        "counts_sum": np.sum(sampling_counts),
        "target_value": target_value,
        "merged": merge_classes,
        "min_px_per_class": min_px,
    }
    return out_array, out_dict


def _sampling_by_selection(
    file: Union[Path, str],
    include: Sequence[int] = None,
    exclude: Optional[Sequence[int]] = None,
):
    """
    Select pixels from a raster file based on inclusion and exclusion criteria.

    Args:
        file (Union[Path, str]): The path to the raster file.
        include (Sequence[int], optional): A sequence of values to include in the selection. Defaults to None.
        exclude (Optional[Sequence[int]], optional): A sequence of values to exclude from the selection. Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[str, Union[str, Number, Sequence[int]]]]:
            - The selected pixels as a numpy array.
            - A dictionary containing the settings used for selection and final results.

    Raises:
        ValueError: If the include and exclude parameters are identical.
    """
    file = Path(file) if isinstance(file, str) else file

    if include == exclude:
        raise ValueError("Include and exclude cannot be identical.")

    raster = load_raster(file)
    raster_array = read_raster(raster, replace_nan=True)

    out_raster, out_settings = _sampling_select_classes(
        array=raster_array,
        include=include,
        exclude=exclude,
    )
    return out_raster, out_settings


def _sampling_by_threshold(
    file: Union[Path, str],
    type: Literal["positives", "negatives"],
    threshold: Number,
    exclude: Optional[Sequence[int]] = None,
):
    """
    Select pixels from a raster based on a given threshold and type.

    Args:
        file (Union[Path, str]): The path to the raster file.
        type (Literal["positives", "negatives"]): The type of pixels to select.
            "positives" selects pixels greater than or equal to the threshold.
            "negatives" selects pixels less than the threshold.
        threshold (Number): The threshold value for selecting pixels.
        exclude (Optional[Sequence[int]]): A sequence of values to exclude from selection.
            Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[str, Union[str, Number, Sequence[int]]]]:
            - The selected pixels as a numpy array.
            - A dictionary containing the settings used for selection and final results.

    Raises:
        ValueError: If the threshold is higher than the maximum value in the raster,
            or if the threshold is lower than the minimum value in the raster,
            or if the threshold is lower than 0.
    """
    file = Path(file) if isinstance(file, str) else file

    raster = load_raster(file)
    raster_array = read_raster(raster, replace_nan=True)

    if threshold > np.nanmax(raster_array):
        raise ValueError(
            f"Threshold {threshold} is higher than the maximum value in the raster."
        )
    elif threshold < np.nanmin(raster_array):
        raise ValueError(
            f"Threshold {threshold} is lower than the minimum value in the raster."
        )
    elif threshold < 0:
        raise ValueError(
            f"Threshold {threshold} is lower than 0, which is not allowed."
        )

    out_raster, out_settings = _sampling_select_classes(
        array=raster_array,
        type=type,
        threshold=threshold,
        exclude=exclude,
    )
    return out_raster, out_settings


# endregion: Sampling functions


# region: Test code

# endregion: Test code
