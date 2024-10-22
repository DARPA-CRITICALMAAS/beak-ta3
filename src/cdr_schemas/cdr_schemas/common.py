from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# Constant for defining that a projection is in pixel coordinates
CRITICALMAAS_PIXEL = "pixel"


class GeomType(str, Enum):
    Point = "Point"
    LineString = "LineString"
    Polygon = "Polygon"


class GeoJsonType(str, Enum):
    Feature = "Feature"
    FeatureCollection = "FeatureCollection"


class ModelProvenance(BaseModel):
    model: str = Field(description="Name of the model used to generate this data")
    model_version: str = Field(
        description="Version of the model used to generate this data"
    )
    model_config = ConfigDict(protected_namespaces=())
    confidence: Optional[Union[float, int]] = Field(
        default=None, description="The prediction confidence of the model"
    )
