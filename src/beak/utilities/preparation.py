import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.impute import SimpleImputer
from typing import List, Tuple, Union, Optional
from numbers import Number


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
    columns: List[str],
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
    if isinstance(data, np.ndarray):
        data_imputed = imputer.fit_transform(data)
    elif isinstance(data, pd.DataFrame) or isinstance(data, gpd.GeoDataFrame):
        data_imputed[columns] = imputer.fit_transform(data_imputed[columns])
    return data_imputed
