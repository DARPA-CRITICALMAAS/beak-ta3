import os
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import urllib3
import zipfile
import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pathlib import Path
from tqdm import tqdm
from pandas import json_normalize
from numbers import Number

from beartype.typing import Union, Optional, Tuple, List
from beak.experimental.io import check_path, data_folder
from beak.experimental.misc import replace_invalid_characters
from beak.experimental.conversions import create_binary_raster


def _get_cma(
    json: Union[str, Path],
    target="cma",
    payloads: str="payload"
) -> pd.DataFrame:
    """
    Extracts CMA data from a JSON file.

    Args:
      json (Union[str, Path]): The path to the JSON file containing CMA data.
      payloads (str): The name of the payload column containing the CMA data.

    Returns:
      pd.DataFrame: A DataFrame containing the CMA information.
    """
    data = pd.read_json(json)
    data = data.loc[target]
    data = data[payloads]

    return data


def _get_evidence_layers(
    json: Union[str, Path],
    target="evidence_layers",
    payloads: str="payload"
) -> pd.DataFrame:
    """
    Extracts evidence layers from a JSON file.

    Args:
      json (Union[str, Path]): The path to the JSON file containing evidence layers.
      payloads (str): The name of the payload column containing the evidence layers.

    Returns:
      pd.DataFrame: A DataFrame containing the evidence layer information.
    """
    data = pd.read_json(json)
    data = data.explode(payloads)
    data = data.loc[target]
    data = json_normalize(data[payloads])

    return data


def _download_cdr_files(
    download_url: Union[str, Path],
    download_file_path: Union[str, Path],
    verify_ssl: bool,
):
    time.sleep(1)
    download_url = str(download_url)

    check_path(download_file_path.parent)
    response = requests.get(download_url, verify=verify_ssl)
    with open(download_file_path, 'wb') as file:
        file.write(response.content)


def create_template_raster(
    cma: dict,
    path: Union[str, Path],
    snap_to: Optional[Tuple[Number, Number]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Creates a template raster using CMA data.

    Args:
      cma (dict): The CMA data.
      path (Union[str, Path]): The path to save the template raster.
      snap_to (Optional[Tuple[Number, Number]]): The geographic x and y coordinates to snap the template raster to.
    """
    path = Path(path)

    extent = cma["extent"]
    crs = cma["crs"]
    resolution = cma["resolution"]

    polygon = gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.shape(extent)]
        },
        crs=crs
    )

    out_array, out_meta = create_binary_raster(
        geodataframe=polygon,
        resolution=resolution,
        all_touched=False,
        same_shape=True,
        fill_negatives=False,
        return_meta=True,
        snap_to_origin=snap_to,
        out_file=path
    )

    return out_array, out_meta


def _zip_file_extract_core(
    file_path: Union[str, Path],
    extract_to: Union[str, Path],
):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def _zip_file_extract_run(
    extract_zip: bool,
    file_path: Union[str, Path],
    extract_to: Union[str, Path],
):
    if extract_zip and file_path.suffix == ".zip":
        file_path = Path(file_path)

        _zip_file_extract_core(
            file_path,
            extract_to
        )


def download_evidence_layers(
    data: pd.DataFrame,
    cma: dict,
    base_folder: Union[str, Path]=data_folder(),
    custom_cma_id: Optional[str] = None,
    extract_zip: bool = True,
    verify_ssl: bool = True,
    ignore: Optional[List[str]] = None,
):
    if custom_cma_id is None:
        cma_id = cma["cma_id"]
    else:
        cma_id = custom_cma_id

    cma_id = replace_invalid_characters(cma_id)
    base_folder = base_folder / "cdr" / cma_id

    for index, row in tqdm(data.iterrows(), total=len(data)):
        transform_methods = row["transform_methods"]
        title = row["title"]
        layer_prefix = row["data_source.evidence_layer_raster_prefix"]
        file_format = row["data_source.format"]
        description = row["data_source.description"]
        reference_url = row["data_source.reference_url"]
        data_type = row["data_source.type"]
        resolution = row["data_source.resolution"]
        download_url = row["data_source.download_url"]
        derivative = row["data_source.derivative_ops"]
        category = row["data_source.category"]
        subcategory = row["data_source.subcategory"]
        source_id = row["data_source.data_source_id"]
        ignored_urls = []

        download_folder = base_folder / category / subcategory
        download_file = download_folder / str(title + Path(download_url).suffix)

        if (
            (ignore is not None and title not in ignore) and
            (not os.path.exists(download_file) and download_url != "")
        ):

            check_path(download_folder)

            _download_cdr_files(
                download_url=download_url,
                download_file_path=download_file,
                verify_ssl=verify_ssl
            )

            _zip_file_extract_run(
                extract_zip=extract_zip,
                file_path=download_file,
                extract_to=download_folder
            )
        else:
            ignored_urls.append(download_url)

        if ignored_urls:
            print("Following URLs where not downloaded:")
            for url in ignored_urls:
                print(f"{title}: {url}")
