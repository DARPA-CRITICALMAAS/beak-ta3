import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from sklearn.impute import SimpleImputer
from beartype.typing import List, Tuple, Union, Optional, Literal
from numbers import Number

from beak.utilities.focal import (
    _create_local_buffer,
)


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
    target_locations = np.isnan(array).any(axis=0)
    out_array = array[:, ~target_locations]

    if transpose is True:
        out_array = np.transpose(out_array)

    return out_array


def create_hard_buffer_around_labels(
    array: np.ndarray,
    radius: int = 1,
    shape: Literal["square", "circle"] = "circle",
    positive_label_value: int = 1,
    buffer_value: Optional[Number] = None,
) -> np.ndarray:
    """
    Extends the positive values in the given array.

    A: If the buffer value is not specified, the buffer value will be the same as the positive labels value.
    B: If the buffer value is specified, elements around the positive labels will be replaced with this value.

    For extending positive labels, use option A.
    For excluding negative labels, use option B with values such as np.nan oder -1.

    Args:
        array (np.ndarray): The array to be extended.
        radius (int): The radius of the buffer (1 for a 3x3 window with 9 pixels, size is n*2 + 1). Defaults to 1.
        shape (Literal["square", "circle"]): The shape of the buffer. Defaults to "square".
        selected_value (int): The value to be extended. Defaults to 1.
        buffer_value (Optional[Number]): The value to be used for the buffer. Defaults to np.nan

    Returns:
        np.ndarray: The extended label's array.
    """
    out_array = np.copy(array)
    out_array = _create_local_buffer(
        array=array,
        radius=radius,
        shape=shape,
        target_value=positive_label_value,
    )

    if buffer_value is not None:
        out_array = np.where(out_array == positive_label_value, buffer_value, out_array)
        out_array = np.where(
            array == positive_label_value, positive_label_value, out_array
        )

    return out_array


# What we need
# 1: A BMU Cluster map with correlated labels (input as path to a file)
# 5: Consider positive and negative label selection (easier for negatives, only one class)

# 2: A function to select the the BMU clusters from the map based on
#    a) the number of labels in a certain cluster
#    b) the percentile of labels in a certain cluster based on the total number of labels in the cluster
def _sampling_select_clusters(array: np.ndarray, threshold: Number, type: Literal["positives", "negatives"]):
    unique_values = np.unique(array)
    cluster_selection = np.nanquantile(unique_values, threshold, method="nearest") if threshold < 1 else threshold

    if type == "positives":
        out_array = np.where(array < cluster_selection, np.nan, array)        
    elif type == "negatives":
        out_array = np.where(array < cluster_selection, array, np.nan)

    print(f"threshold: {cluster_selection}")
    return out_array


# 3: A function to select the random points based on the calculated number of points
#   a) equally distributed among all clusters (equal)
#   b) distributed depending on the number of pixels available for each class (relative)
def _sampling_select_random_points():
    return


# 4: A testing function that checks if enough pixels are available
def _sampling_check_number_of_pixels():
    return 


def sampling_from_clusters(file: Union[Path, str], threshold: Number, type: Literal["positives", "negatives"]):
    if isinstance(file, str):
        file = Path(file)

    raster = rasterio.open(file)
    raster_array = raster.read()
    print(np.unique(raster_array, return_counts=True))
    print(raster.nodata)
    raster_array = np.where(raster_array == raster.nodata, np.nan, raster_array)

    if threshold >= np.nanmax(raster_array):
        raise ValueError(f"Threshold {threshold} is higher than the maximum value in the raster.")
    if threshold <= 0:
        raise ValueError(f"Threshold {threshold} is lower or equal to 0.")

    out_array = _sampling_select_clusters(raster_array, threshold, type)
    return out_array


# region: Test code
if __name__ == "__main__":
    # User inputs
    folder = Path("S:/Projekte/20230082_DARPA_CriticalMAAS_TA3/Bearbeitung/GitHub/beak-ta3/experiments/hackathon_9m_related/03_cma/cobalt_nickel_upper_midwest/som/models/")
    model_config = "BASELINE_BISON"

    model_run_random = "SOM_BASELINE_BISON_F28_X50_Y50_CMAX50_20240612-134211"
    model_run_pca = "SOM_BASELINE_BISON_F28_X50_Y50_CMAX50_20240612-135610"
    model_run = model_run_random

    cluster_map = folder / model_config / model_run / "exports" / "GeoTIFF" / "BMU_BMU_label_count.tif"

    # Load data
    file = cluster_map

    types = ["positives", "negatives"]
    for type in types:
        print("\n")
        print("Type: " + type)
        output = sampling_from_clusters(file, threshold=3, type=type)
        print(np.unique(output, return_counts=True))

# endregion: Test code