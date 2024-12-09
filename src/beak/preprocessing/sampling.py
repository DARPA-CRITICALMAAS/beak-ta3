import numpy as np

from imblearn.over_sampling import RandomOverSampler
from typing import Tuple, Union


def select_random_samples(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = 20,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    # TODO: Docstring goes here
    """
    np.random.seed(random_seed)

    X_positives = X[y == 1]
    X_negatives = X[y == 0]

    num_negatives = min(X_positives.shape[0] * multiplier, X_negatives.shape[0])

    negative_indices = np.random.choice(X_negatives.shape[0], num_negatives, replace=False)
    X_samples_negative = X_negatives[negative_indices]

    X_samples = np.vstack((X_positives, X_samples_negative))
    y_samples = np.concatenate([np.ones(X_positives.shape[0]), np.zeros(num_negatives)])

    return X_samples, y_samples


def random_oversampling(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Union[int, float] = 0.25,
    random_seed: int = 42,
    shrinkage: Union[int, float] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    # TODO: Docstring goes here
    """
    if sampling_strategy > 0:
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_seed, shrinkage=shrinkage)
        X_samples, y_samples = ros.fit_resample(X, y)
    else:
        X_samples, y_samples = X, y

    return X_samples, y_samples
