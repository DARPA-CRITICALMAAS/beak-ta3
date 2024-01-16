import os
import re
from pathlib import Path
from typing import Optional


def replace_invalid_characters(string: str) -> str:
    """
    Replace invalid characters in a string with an underscore.
    Multiple consecutive underscores will be cropped to one.

    Args:
        string (str): Input string.

    Returns:
        str: String with replaced characters.
    """
    string = re.sub(r"[ /().,]", "_", string)
    string = re.sub(r"(_+)", "_", string)
    return string


def create_tree(
    path: Optional[str] = None,
    parent: bool = False,
    depth: int = 0,
    parent_prefix: str = "",
    is_last: bool = False,
    excluded_folders: list = ["eis_toolkit", "__pycache__"],
):
    """Create a tree structure of a given directory."""
    path = Path.cwd() if path is None else Path(path)
    path = path.parent if parent is True and depth == 0 else path

    folders = [
        entry
        for entry in os.listdir(path)
        if os.path.isdir(os.path.join(path, entry))
        and entry not in excluded_folders
        and not entry.startswith((".", "_"))
    ]

    # Sort folders for consistency
    folders.sort()

    tree = ""
    entry_prefix = "└── " if is_last else "├── "

    # Display the current directory
    tree += parent_prefix + entry_prefix + os.path.basename(path) + "/\n"

    # Prepare the prefix for subdirectories
    prefix = parent_prefix + ("    " if is_last else "│   ")

    if depth < 2:
        for i, folder in enumerate(folders):
            full_path = os.path.join(path, folder)

            # Check if it is the last entry in the current directory
            last_entry = i == len(folders) - 1

            tree += create_tree(
                full_path, parent, depth + 1, prefix, last_entry, excluded_folders
            )

    return tree
