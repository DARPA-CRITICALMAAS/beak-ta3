import pandas as pd
import os
import sys
import requests
import zipfile
from pathlib import Path
from beartype.typing import List, Tuple, Optional, Sequence, Union, Literal


def _get_data_folder(folder_name: str = "beak.data") -> Path:
    """
    Returns the path to the specified data folder.
    """
    if sys.version_info < (3, 9):
        from importlib_resources import files
    else:
        from importlib.resources import files

    return files(folder_name)


def _extract_rows(
    data: pd.DataFrame,
    attribute: str,
) -> List:
    """
    Extract specific information from the provided data frame.
    """
    extractions = []
    attribute = "data_source" + "." + attribute

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
    Filters out files from the provided list of files.
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
    Downloads a file from the provided URL to the specified download folder.
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
    Creates a zip file from a list of files.
    """
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in file_list:
            file = Path(file)
            archive.write(filename=file, arcname=file.name)