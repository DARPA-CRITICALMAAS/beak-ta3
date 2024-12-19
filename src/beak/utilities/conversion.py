import numpy as np
from typing import List, Union

def convert_dtypes(data: List[Union[np.ndarray, None]], dtype: str = "float32") -> List[Union[np.ndarray, None]]:
    """
    Converts data to the specified dtype.

    Args:
        data: The input data.
        dtype: The desired dtype. Defaults to "float32".

    Returns:
        List of the converted data with the specified dtype.
    """
    return [element.astype(dtype) if isinstance(element, np.ndarray) else None for element in data]


def expand_dims(data: List[Union[np.ndarray, None]], axis: int = 1) -> List[Union[np.ndarray, None]]:
    """
    Expands the dimensions of the input data.

    Returns None if the input is element is None.

    Args:
        data: The input array.
        axis: The axis along which to expand the dimensions. Defaults to 1.

    Returns:
        List containing the expanded arrays.
    """
    return [np.expand_dims(element, axis=axis) if isinstance(element, np.ndarray) else None for element in data]
