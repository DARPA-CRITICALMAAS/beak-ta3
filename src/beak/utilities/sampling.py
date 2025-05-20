import pyproj
import rasterio
import numpy as np
import geopandas as gpd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from typing import Tuple, Union, Optional

from beak.utilities.vector_processing import create_geodataframe_from_points


def select_negative_samples(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = 20,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples negative samples based on the number of positives with a given multiplier.

    Special cases:
        - Uses 0 for negatives by default. Custom negatives can be used by providing labels with -1.
        - If the multiplier equals 0, only positives will remain.
        - If the multiplier is less than 0, all negatives will be used (no sampling).
        - If the number of negatives available is less than the desired number, all negatives will be used.

    Returns:
        The sampled data and labels for the negative samples.
    """
    np.random.seed(random_seed)

    X_positives = X[y == 1]
    negative_label = -1 if np.any(y == -1) else 0
    X_negatives = X[y == negative_label]

    if multiplier >= 0:
        max_negatives = X_negatives.shape[0]
        fraction_negatives = X_positives.shape[0] * multiplier
        num_negatives = min(X_negatives.shape[0], fraction_negatives)

        negative_indices = np.random.choice(max_negatives, num_negatives, replace=False)
        X_sampled_negatives = X_negatives[negative_indices]

        X_samples = np.vstack((X_positives, X_sampled_negatives))
        y_samples = np.concatenate([np.ones(X_positives.shape[0]), np.zeros(num_negatives)])
    else:
        X_samples, y_samples = X, np.where(y == 1, 1, 0)

    return X_samples, y_samples


def random_oversampling(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.25,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversampling of positive labels using RandomOverSampler.

    Sampling will not be performed if:
        - The sampling strategy is 0.
        - The number of negative samples is less than or equal to the number of positive samples.

    Args:
        X: Input data.
        y: Input labels.
        sampling_strategy: Sampling strategy for positive labels. Default to 0.25.
        random_seed: Random seed for reproducibility. Default to 42.

    Returns:
        Oversampled data and labels.
    """
    negatives_count = np.count_nonzero(y == 0)
    positives_count = np.count_nonzero(y == 1)

    oversample = (
        sampling_strategy > 0
        and negatives_count * sampling_strategy > positives_count
    )

    if oversample:
        ros = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_seed,
            shrinkage=0,
        )
        X_samples, y_samples = ros.fit_resample(X, y)
    else:
        X_samples, y_samples = X, y

    return X_samples, y_samples


def select_train_and_test_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    Split input data into test and training sets.

    Splitting is not performed if: train size is 0 or 1.
    None elements are simply being passed through.

    Args:
        X: Input data.
        y: Input labels.
        train_size: Proportion of the dataset to include in the train split. Default to 1 (full dataset).
        random_seed: Random seed for reproducibility. Default to 42.

    Returns:
        Training and test data splits.
    """
    if 0 < train_size < 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_seed)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    return X_train, X_test, y_train, y_test


def extract_train_test_locations(
    X: Optional[np.ndarray],
    y: Optional[np.ndarray],
    crs: Optional[Union[str, rasterio.crs.CRS, pyproj.crs.CRS]],
) -> Tuple[Union[np.ndarray, None], Union[gpd.GeoDataFrame, None]]:
    """
    Extracts the training locations from the input data.

    Args:
        X: Input data with last two columns as the longitude/latitude coordinates.
        y: Input labels.
        crs: Coordinate reference system for the locations.

    Returns:
        Data with coordinates removed, and the locations themselves.
    """
    if all(isinstance(data, np.ndarray) for data in (X, y)):
        locations = gpd.GeoDataFrame(
            data=np.column_stack((y, X[:, -2:])),
            geometry=gpd.points_from_xy(X[:, -2], X[:, -1]),
            columns=["label", "longitude", "latitude"],
            crs=crs
        )
        X = X[:, :-2]
    else:
        locations = gpd.GeoDataFrame()

    return X, locations
