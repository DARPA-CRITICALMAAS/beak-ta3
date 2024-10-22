from typing import List, Optional, Union

from pydantic import BaseModel, Field

from cdr_schemas.area_extraction import AreaType
from cdr_schemas.cdr_responses.features import ProjectedFeature
from cdr_schemas.features.polygon_features import Polygon


class AreaExtractionResponse(BaseModel):
    area_extraction_id: str = Field(default="", description="Area extraction id")
    cog_id: str = Field(default="", description="Cog id")
    reference_id: Union[str, None] = Field(
        default=None, description="Legend id of older version of this legend item."
    )
    px_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the item.
                    Format is expected to be [x1,y1,x2,y2].""",
    )
    px_geojson: Optional[Polygon]
    system: str = Field(default="", description="System that published this item")
    system_version: str = Field(
        default="", description="System version that published this item"
    )
    model_id: str = Field(
        default="", description="Model id for the model used to generate this item"
    )
    validated: Optional[bool] = Field(default=None, description="Validated by human")
    confidence: Optional[float] = None
    category: AreaType = Field(
        ...,
        description="""
            The type of area extraction.
        """,
    )
    text: str = Field(
        default="",
        description="""
            The text within the extraction area.
        """,
    )
    projected_feature: List[ProjectedFeature] = Field(
        default_factory=list,
        description="""
            List of projected versions of this feature. Probably will only be one result.
        """,
    )
