import os
import shutil
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

def copy_relevant_files(
    input_folders: List[Path],
    output_folder: Path,
    file_extensions: Optional[List[str]] = None
):
    """
    Copies .py and .ipynb files from a list of folders to a target root folder while keeping the relative structure.

    Args:
        input_folders (List[Path]): List of folders to search for files.
        output_folder (Path): The target root folder to save the files.
        file_extensions (List[str]): List of file extensions to copy. Defaults to ['.py', '.ipynb'].

    Returns:
        None
    """
    if file_extensions is None:
        file_extensions = [".py", ".ipynb"]

    # Create the target root folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through each folder and copy relevant files to the target folder
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    source_file_path = Path(root) / file
                    relative_path = source_file_path.relative_to(folder)
                    destination_file_path = output_folder / relative_path

                    # Create the destination directory
                    destination_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy
                    shutil.copy2(source_file_path, destination_file_path)
                    print(f"Copied {source_file_path} to {destination_file_path}")


def copy_files_excluding(
    input_folders: List[Path],
    output_folder: Path,
    exclude_extensions: Optional[List[str]] = None,
    exclude_folders: Optional[List[Path]] = None,
    verbose: int = 0,
):
    """
    Copies files from a list of folders to a target root folder while keeping the relative structure,
    excluding specific file extensions and folders.

    Args:
        input_folders (List[Path]): List of folders to search for files.
        output_folder (Path): The target root folder to save the files.
        exclude_extensions (List[str]): List of file extensions to exclude.
            Defaults to None.
        exclude_folders (List[Path]): List of folders to exclude.
            Defaults to None.
        verbose (int): Verbosity level. 0: Silent, 1: Full,

    Returns:
        None
    """
    if exclude_extensions is None:
        exclude_extensions = [
            ".aux", ".xml", ".cpg", ".ovr", ".dbf",
            ".prj", ".sld", ".shp", ".shx", ".tfw",
            ".tif", ".tiff", ".png", ".pdf", ".jpg",
            ".jpeg", ".zip", ".7z", ".rar", ".7zip",
            ".gz", ".bz2", ".qgz",
        ]

    if exclude_folders is None:
        exclude_folders = [
            "models", "data", "logs", "results", ".git",
            ".idea", "beak.egg-info", "__pychache__", "saved_models",
            "BAK",
        ]

    # Create the target root folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through each folder and copy relevant files to the target folder
    for folder in input_folders:
        for root, dirs, files in os.walk(folder):
            # Skip excluded folders
            dirs[:] = [name for name in dirs if name not in exclude_folders]

            for file in tqdm(files):
                if not any(file.endswith(ext) for ext in exclude_extensions):
                    source_file_path = Path(root) / file

                    relative_path = source_file_path.relative_to(folder.parent)
                    destination_file_path = output_folder / relative_path

                    # Create the destination directory
                    destination_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy
                    shutil.copy2(source_file_path, destination_file_path)

                    if verbose == 1:
                        print(f"Copied {source_file_path} to {destination_file_path}")

if __name__ == "__main__":
    in_folders = [Path("S:/Projekte/20230082_DARPA_CriticalMAAS_TA3/Bearbeitung/GitHub/beak-ta3")]
    out_folder = Path("S:/Temp/Projekte/Backups/Experiments/code_before_refactoring_to_experimental/")

    copy_files_excluding(in_folders, out_folder)

