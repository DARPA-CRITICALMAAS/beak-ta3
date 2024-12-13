from typing import Optional, Tuple, List


def build_network_architecture(
    depth_per_network: int,
    minimum_width: int,
    core_units: Optional[List[int]] = None,
    head_units: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Calculate the base and doubled topologies for a given depth and width.

    Args:
        depth_per_network: The number of layers in the network.
        minimum_width: The number of neurons in the last layer.
        core_units: The number of neurons in the core units. Defaults to None.
        head_units: The number of neurons in the head units. Defaults to None.

    Returns:
        Two lists, containing the topology for the core and head units, respectively.
    """
    if core_units is None and head_units is None:
        head_units = [2 ** (minimum_width + i) for i in range(depth_per_network)][::-1]
        core_units = [max(head_units) * (2 ** (i + 1)) for i in range(depth_per_network)][::-1]

    return core_units, head_units
