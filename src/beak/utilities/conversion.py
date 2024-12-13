import numpy as np
import tensorflow as tf

from typing import List, Union

def convert_dtypes(data: List[Union[np.ndarray, None]], dtype: str = "float32") -> List[Union[np.ndarray, None]]:
    """
    # TODO: Docstring goes here.
    """
    return [element.astype(dtype) if isinstance(element, np.ndarray) else None for element in data]


def expand_dims(data: List[Union[np.ndarray, None]], axis: int = 1) -> List[Union[np.ndarray, None]]:
    """
    # TODO: Docstring goes here.
    """
    return [np.expand_dims(element, axis=axis) if isinstance(element, np.ndarray) else None for element in data]
