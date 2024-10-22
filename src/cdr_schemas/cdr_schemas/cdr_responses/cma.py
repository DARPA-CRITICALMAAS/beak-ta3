from datetime import datetime
from typing import List, Optional, Union

from geojson_pydantic import MultiPolygon
from pydantic import BaseModel


class BestBounds(BaseModel):
    type: str
    coordinates: List[List[List[Union[float, int]]]]


class CMA(BaseModel):
    cma_id: str
    crs: str
    mineral: str
    download_url: str
    extent: Optional[MultiPolygon]
    resolution: List[int]
    description: Optional[str] = ""
    creation_date: Optional[datetime]
