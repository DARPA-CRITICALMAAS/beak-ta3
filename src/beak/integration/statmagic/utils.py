import pandas as pd
import os
import sys
import requests
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal, Any


def _get_data_folder(package_name = "beak", folder_name: str = "data") -> Any:
    """
    Obtain the path to the specified data folder.

    Args:
        folder_name: The name of the data folder. Default is "beak.data".

    Returns:
        The relative path to the specified data folder.
    """
    if sys.version_info < (3, 9):
        from importlib_resources import files
    else:
        from importlib.resources import files

    return files(package_name) / folder_name


def _extract_rows(
    data: pd.DataFrame,
    attribute: str,
) -> List:
    """
    Extract specific information from the provided data frame.

    Args:
        data: The DataFrame containing the data.
        attribute: The column name to extract information from for each row.

    Returns:
        A list containing the extracted information.
    """
    extractions = []

    for index, row in data.iterrows():
        information = row[attribute]
        extractions.append(information)

    return extractions


def create_file_list(
    folder: Union[str, Path],
    file_suffix: Union[str, Tuple, None] = (".tif", ".tiff"),
    file_prefix: Union[str, Tuple, None] = None,
) -> List[str]:
    """
    Create a list of files in the specified folder with the given extensions.

    Args:
        folder: The folder path to search for files.
        file_suffix: Sequence of file extensions to include. Defaults to [".tif", ".tiff"].
        file_prefix: Prefixes of files to filter. Defaults to None.

    Returns:
        A list of file paths found.
    """
    folder = Path(folder)
    file_list = [str(file) for file in folder.glob("*")]

    return _filter_files(
        file_list,
        file_suffix=file_suffix,
        file_prefix=file_prefix,
    )


def _filter_files(
    file_list: List[str],
    file_suffix: Union[str, Tuple, None],
    file_prefix: Union[str, Tuple, None],
) -> List[str]:
    """
    Filter files from the provided list of files.

    Args:
        file_list: List of file paths.
        file_suffix: Suffixes of files to filter.
        file_prefix: Prefixes of files to filter.

    Raises:
        AssertionError: If no files are found.

    Returns:
        List of file paths that match the specified criteria.
    """
    suffix_match, prefix_match = True, True
    filtered_files = []

    for file in file_list:
        file_name = str(
            Path(file).name
        ).lower()

        if file_suffix is not None:
            file_suffix = tuple(ext.lower() for ext in file_suffix) if isinstance(file_suffix, tuple) else file_suffix.lower()
            suffix_match = file_name.endswith(file_suffix)

        if file_prefix is not None:
            file_prefix = tuple(prefix.lower() for prefix in file_prefix) if isinstance(file_prefix, tuple) else file_prefix.lower()
            prefix_match = file_name.startswith(file_prefix)

        if suffix_match and prefix_match:
            filtered_files.append(file)

    assert filtered_files, "No files found."
    return filtered_files


def _download_cdr_files(
    download_urls: List[str],
    download_folder: Path,
    verify_ssl: bool,
) -> List[str]:
    """
    Download files from the provided list of URLs to a specified folder.

    Args:
        download_urls: List of URLs from which to download files.
        download_folder: Path to the folder where the files will be downloaded.
        verify_ssl: Whether to verify SSL certificates during the download.

    Returns:
        List of file paths that were downloaded.
    """
    download_folder.mkdir(parents=True, exist_ok=True)

    file_list = []
    for url in download_urls:
        url_name = Path(url).name
        download_file_path = download_folder / url_name

        if not os.path.exists(download_file_path):
            response = requests.get(url, verify=verify_ssl)
            with open(download_file_path, "wb") as file:
                    file.write(response.content)

        file_list.append(
            str(download_file_path)
        )

    return file_list


def create_zip_from_files(
    file_list: List[Union[str, Path]],
    archive_path: Union[str, Path]
):
    """
    Create a zip file from a list of files.

    Args:
        file_list: The list of input files to be saved.
        archive_path: The path to the output zip file.

    Returns:
        None
    """
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in file_list:
            file = Path(file)
            archive.write(filename=file, arcname=file.name)


def delete_files(file_list: List[str]) -> None:
    """
    Delete files from the specified list.

    Args:
        file_list: List of file paths to delete.

    Returns:
        None
    """
    for file in file_list:
        try:
            os.remove(file)
        except:
            print(f"Failed to delete file: {file}")


def _filter_layers_from_payload(
    payload: pd.DataFrame,
    label_column: Optional[str],
    filter_labels: bool = False
) -> pd.DataFrame:
    """
    Filter evidence layers from configuration file.

    Args:
        payload: DataFrame containing the payload.
        label_column: Column name containing labels.
        filter_labels: Whether to filter layers based on labels. Defaults to False.

    Returns:
        Filtered dataframe containing evidence layers or label file.
    """
    layers = payload["event"]["evidence_layers"]
    layers = pd.json_normalize(layers)

    if label_column in layers.columns:
        layers = layers[
            layers[label_column] == filter_labels
        ]

    return layers