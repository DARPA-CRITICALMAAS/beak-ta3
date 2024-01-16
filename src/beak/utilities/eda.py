import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union, List, Literal

# References
# Some non-trivial functionalities were adapted from other sources.
# The original sources are listed below and referenced in the code as well.
#
# EIS toolkit:
# GitHub repository https://github.com/GispoCoding/eis_toolkit under EUPL-1.2 license.


def get_outliers_zscore(
    data: pd.DataFrame, column: str, threshold: np.number = 3.0
) -> pd.DataFrame:
    """
    Get outliers based on the z-score using scikit-learn.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to calculate z-scores and identify outliers.
        threshold (np.number): The threshold value for identifying outliers. Defaults to 3.0.

    Returns:
        pd.DataFrame: A DataFrame containing the outliers based on the z-score.

    """
    # Extract the column data
    column_data = data[[column]]

    # Use StandardScaler to calculate z-scores
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(column_data)

    # Identify outliers and return as DataFrame
    outliers = pd.DataFrame(data.loc[np.abs(z_scores) > threshold, column])

    return outliers


def get_outliers_iqr(
    data: Union[pd.DataFrame, np.ndarray],
    column: Optional[str] = None,
    threshold: np.number = 1.5,
) -> pd.Series:
    """
    Get outliers based on the IQR method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column name to calculate outliers for.
        threshold (np.number): The threshold value for outlier detection. Defaults to 1.5.

    Returns:
        pd.Series: A Series containing the outliers.

    """
    if isinstance(data, pd.DataFrame):
        # Check if column is provided
        assert column is not None

        # Extract the column data
        values = data[column].values
    else:
        values = data

    # Calculate quantiles
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Identify outliers
    outliers = values[(values < lower_bound) | (values > upper_bound)]

    return lower_bound, upper_bound, outliers


def clip_outliers(
    data: pd.DataFrame, columns: List[str], threshold: np.number = 1.5
) -> pd.DataFrame:
    """
    Clip outliers in the specified columns of a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (List[str]): The list of column names to clip outliers.
        threshold (np.number): The threshold value for clipping outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers clipped in the specified columns.
    """
    data_cleaned = data.copy()

    for column in columns:
        lower_bound, upper_bound, _ = get_outliers_iqr(
            data_cleaned[column].values, threshold=threshold
        )
        data_cleaned.loc[data_cleaned[column] < lower_bound, column] = lower_bound
        data_cleaned.loc[data_cleaned[column] > upper_bound, column] = upper_bound

    return data_cleaned
