from typing import List, Optional, Union

from pydantic import BaseModel, Field

from cdr_schemas.cdr_responses.features import (
    LineExtractionResponse,
    PointExtractionResponse,
    PolygonExtractionResponse,
)
from cdr_schemas.features.polygon_features import Polygon


class LegendItemResponse(BaseModel):
    legend_id: str = Field(default="", description="CDR legend id")
    abbreviation: str = Field(default="", description="Abbreviation of legend item")
    description: str = Field(default="", description="Legend item description")
    color: str = Field(default="", description="Color")
    reference_id: Union[str, None] = Field(
        default=None, description="Legend id of older version of this legend item."
    )
    label: str = Field(default="", description="Label of legend item")
    pattern: str = Field(
        default="",
        description="If category of type polygon this can be filled in with pattern type",
    )
    px_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the item.
                    Format is expected to be [x1,y1,x2,y2].""",
    )
    px_geojson: Optional[Polygon]
    cog_id: str = Field(default="", description="Cog id")
    category: str = Field(
        default="", description="Category of legend item. Polygon, point, or line."
    )
    system: str = Field(default="", description="System that published this item")
    system_version: str = Field(
        default="", description="System version that published this item"
    )
    model_id: str = Field(
        default="", description="Model id for the model used to generate this item"
    )
    validated: Optional[bool] = Field(default=None, description="Validated by human")
    confidence: Optional[float] = None
    map_unit_age_text: list = Field(default_factory=list, description="Age of map unit")
    map_unit_lithology: list = Field(
        default_factory=list, description="Map unit lithology"
    )
    map_unit_b_age: Optional[float] = None
    map_unit_t_age: Optional[float] = None

    point_extractions: List[PointExtractionResponse] = Field(
        default_factory=list,
        description="Optionally added point extractions associated with this legend item",
    )
    polygon_extractions: List[PolygonExtractionResponse] = Field(
        default_factory=list,
        description="Optionally added polygon extractions associated with this legend item",
    )
    line_extractions: List[LineExtractionResponse] = Field(
        default_factory=list,
        description="Optionally added line extractions associated with this legend item",
    )
