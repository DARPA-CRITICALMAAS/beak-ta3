import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    data: pd.DataFrame, column: str, threshold: np.number = 1.5
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
    # Extract the column data
    column_data = data[column]

    # Calculate IQR
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Identify outliers
    outliers = pd.DataFrame(
        data.loc[(column_data < lower_bound) | (column_data > upper_bound), column]
    )
    return outliers
