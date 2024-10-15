import pandas as pd
import os
import sys
import requests
from pathlib import Path
from beartype.typing import List, Tuple, Optional


def _get_data_folder(folder_name: str = "beak.data") -> Path:
    """
    Returns the path to the specified data folder.
    """
    if sys.version_info < (3, 9):
        from importlib_resources import files
    else:
        from importlib.resources import files

    return files(folder_name)


def _select_data(
    data: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Selects a specific group from the provided DataFrame.
    """
    return data.loc[target]


def _extract_payload(
    data: pd.DataFrame,
    target: Optional[str],
    normalize: Optional[bool],
) -> pd.DataFrame:
    """
    Takes a DataFrame and extracts the payload information.
    """
    if target is not None:
        data = _select_data(
            data,
            target,
        )

    payload = data["payload"]
    payload = pd.json_normalize(payload) if normalize is True else payload

    return payload


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


def _filter_tif_files(
    file_list: List[str],
    file_extensions: Tuple[str] = (".tif", ".tiff"),
) -> List[str]:
    """
    Filters out TIF files from the provided download URLs.
    """
    filtered_urls = []
    for url in file_list:
        if url.endswith(file_extensions):
            filtered_urls.append(url)

    assert filtered_urls, "No TIF files found in the provided download URLs."
    return filtered_urls


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