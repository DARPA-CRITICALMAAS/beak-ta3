from numbers import Number

import numpy as np
from beartype.typing import Literal, Callable, Optional
from scipy.ndimage import generic_filter, binary_dilation


def _check_filter_size(radius: int):
    """
    Check the filter size and raise exceptions if it does not meet the requirements.

    Args:
        radius: The size of the filter kernel.

    Raises:
        Exception: If the resulting filter radius is too small.
    """
    if radius < 1:
        raise Exception("Radius must be greater than or equal to 1.")


def _get_kernel_size(radius: int) -> tuple[int, int]:
    """
    Calculate the kernel size and on the given radius.

    Args:
        radius: The radius of the filter kernel.

    Returns:
        A tuple containing the calculated size and radius of the kernel.
    """
    size = 1 + (radius * 2)
    return size, radius


def _create_grid(size: int, radius) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a grid of coordinates.

    Args:
        size: The size of the grid.

    Returns:
        A tuple with x and y coordinates of the grid.
    """
    y, x = np.ogrid[-radius : (size - radius), -radius : (size - radius)]
    return x, y


def _basic_kernel(radius: int, shape: Literal["square", "circle"]) -> np.ndarray:
    """
    Generate a basic kernel of a specified size and shape.

    Args:
        radius: The radius of the kernel.
        shape: The shape of the kernel. Can be either "square" or "circle".

    Returns:
        The generated kernel.
    """
    _check_filter_size(radius)
    size, _ = _get_kernel_size(radius)

    if shape == "square":
        kernel = np.ones((size, size))
    elif shape == "circle":
        x, y = _create_grid(size, radius)
        mask = x**2 + y**2 <= radius**2
        kernel = np.zeros((size, size))
        kernel[mask] = 1
    return kernel


def _apply_generic_filter(
    array: np.ndarray, filter_fn: Callable, kernel: np.ndarray, *args
) -> np.ndarray:
    """
    Apply a generic filter to the input array.

    Args:
        array: The input array to be filtered.
        filter_fn: The filter function to be applied.
        kernel: The kernel or footprint to be used for filtering.
        *args: Additional arguments to be passed to the filter function.

    Returns:
        The filtered array.
    """
    return generic_filter(array, filter_fn, footprint=kernel, extra_arguments=args)


def _apply_binary_dilation(
    array: np.ndarray, kernel: np.ndarray, target_value: Number
) -> np.ndarray:
    """
    Apply binary dilation to the input array.

    Args:
        array (np.ndarray): The input array to be dilated.
        kernel (np.ndarray): The kernel or footprint to be used for dilation.
        target_value (Number): The value to be dilated.

    Returns:
        np.ndarray: The dilated array.
    """
    return binary_dilation(array == target_value, structure=kernel)


def _replace_value_in_kernel(window: np.ndarray, target_value: Number) -> Number:
    p_center = window[window.shape[0] // 2]
    mask = np.where(window == target_value, True, False)

    if sum(mask) > 0:
        return target_value
    else:
        return p_center


def create_simple_buffer(
    array: np.ndarray,
    target_value: int = 1,
    radius: int = 1,
    const: Number = np.nan,
) -> np.ndarray:
    """
    Creates a buffer with a constant value around the selected value in a given array.

    Args:
        array (np.ndarray): The array to be buffered.
        target_value (Number): The value to be buffered.
            Defaults to 1.
        radius (int): Number of cells (distance) to the targeted value to be changed.
            Defaults to 1. E.g. radius of 1 will change the 8 cells around the targeted value.
        const (Number): The constant value to be used. Defaults to np.nan.

    Returns:
        np.ndarray: The buffered array.
    """
    target_locations = np.where(array == target_value)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32)

    out_array = np.copy(array)

    for x, y in zip(target_locations[0], target_locations[1]):
        row_range = slice(max(0, x - radius), min(array.shape[0], x + radius + 1))
        col_range = slice(max(0, y - radius), min(array.shape[1], y + radius + 1))
        out_array[row_range, col_range] = const

    return np.where(array == target_value, target_value, out_array)


def _create_local_buffer_generic_version(
    array: np.ndarray,
    radius: int,
    shape: Literal["square", "circle"],
    target_value: int,
) -> np.ndarray:
    """
    Creates a buffer with a constant value around the selected value in a given array.

    Args:
        array (np.ndarray): The array to be buffered.
        radius (int): Number of cells (distance) to the targeted value to be changed.
        shape (Literal["square", "circle"]): The shape of the buffer.
        target_value (int): The value to be buffered.

    Returns:
        np.ndarray: The buffered array.
    """
    kernel = _basic_kernel(radius, shape)
    array = np.squeeze(array) if array.ndim >= 3 else array

    return _apply_generic_filter(
        array,
        _replace_value_in_kernel,
        kernel,
        target_value,
    )


def _create_local_buffer_binary_version(
    array: np.ndarray,
    radius: int,
    shape: Literal["square", "circle"],
    target_value: Number,
) -> np.ndarray:
    """
    Creates a buffer with a constant value around the selected value in a given array.

    Args:
        array (np.ndarray): The array to be buffered.
        radius (int): Number of cells (distance) to the targeted value to be changed.
        shape (Literal["square", "circle"]): The shape of the buffer.
        target_value (Number): The value to be buffered.

    Returns:
        np.ndarray: The buffered array.
    """
    kernel = _basic_kernel(radius, shape)
    array = np.squeeze(array) if array.ndim >= 3 else array

    return binary_dilation(array == target_value, structure=kernel)
