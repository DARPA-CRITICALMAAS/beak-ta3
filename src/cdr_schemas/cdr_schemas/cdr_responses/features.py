from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field

from cdr_schemas.features.line_features import DashType, Line
from cdr_schemas.features.point_features import Point
from cdr_schemas.features.polygon_features import Polygon


class ProjectedFeature(BaseModel):
    cdr_projection_id: str = Field(
        description="CDR projection id used for creating transform"
    )
    feature_type: str = Field(description="Feature type. E.g. polygon, point, line")
    projected_geojson: Optional[Union[Polygon, Point, Line]] = Field(
        description="Projected geojson in EPSG 4326"
    )
    projected_bbox: Optional[Polygon] = Field(default=None, description="Optional bbox")


class PolygonExtractionResponse(BaseModel):
    polygon_id: str = Field(default="", description="CDR polygon id")
    cog_id: str = Field(default="", description="Cog id")
    px_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the item.
                    Format is expected to be [x1,y1,x2,y2].""",
    )
    px_geojson: Polygon
    reference_id: Union[str, None] = Field(
        default=None, description="Polygon id of older version of this polygon."
    )
    confidence: Optional[float] = None
    model_id: str = Field(
        default="", description="CDR model id for the model used to generate this item"
    )
    system: str = Field(default="", description="System that published this item")
    system_version: str = Field(
        default="", description="System version that published this item"
    )
    validated: Optional[bool] = Field(default=None, description="Validated by human")
    legend_id: Optional[str] = Field(
        default=None, description="Associated CDR legend id"
    )
    projected_feature: List[ProjectedFeature] = Field(
        default_factory=list,
        description="""
            List of projected versions of this feature. Probably will only be one result.
        """,
    )
    legend_item: Optional[Any] = Field(
        default=None,
        description="Some CDR endpoints can allow legend item data attached to each feature.",
    )


class PointExtractionResponse(BaseModel):
    point_id: str = Field(default="", description="CDR point id")
    cog_id: str = Field(default="", description="Cog id")
    px_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the item.
                    Format is expected to be [x1,y1,x2,y2].""",
    )
    px_geojson: Point
    dip: Optional[Union[int, None]] = Field(
        default=None, description="Point dip value."
    )
    dip_direction: Optional[Union[int, None]] = Field(
        default=None, description="Point dip direction value."
    )
    reference_id: Union[str, None] = Field(
        default=None, description="Point id of older version of this point."
    )
    confidence: Optional[float] = None
    model_id: str = Field(
        default="",
        description="""
            Model id associated with the model and version used to generate this item
        """,
    )
    system: str = Field(default="", description="System that published this item")
    system_version: str = Field(
        default="", description="System Version that published this item"
    )
    validated: Optional[bool] = Field(default=None, description="Validated by human")
    legend_id: Optional[str] = Field(default=None, description="Associated legend id")
    projected_feature: List[ProjectedFeature] = Field(
        default_factory=list,
        description="""
            List of projected versions of this feature. Probably will only be one result.
        """,
    )
    legend_item: Optional[Any] = Field(
        default=None,
        description="Some CDR endpoints can allow legend item data attached to each feature",
    )


class LineExtractionResponse(BaseModel):
    line_id: str = Field(default="", description="CDR line id")
    cog_id: str = Field(default="", description="Cog id")
    px_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the item.
                    Format is expected to be [x1,y1,x2,y2].""",
    )
    px_geojson: Line
    dash_pattern: DashType = Field(
        default=DashType.none, description="Dash pattern of line"
    )
    symbol: str = Field(default="", description="Symbol on line")
    reference_id: Union[str, None] = Field(
        default=None, description="Line id of older version of this line."
    )
    confidence: Optional[float] = None
    model_id: str = Field(
        default="", description="model id for the model used to generate this item"
    )
    system: str = Field(default="", description="System that published this item")
    system_version: str = Field(
        default="", description="System version that published this item"
    )
    validated: Optional[bool] = Field(default=None, description="Validated by human")
    legend_id: Optional[str] = Field(default=None, description="Associated legend id")
    projected_feature: List[ProjectedFeature] = Field(
        default_factory=list,
        description="""
            List of projected versions of this feature. Probably will only be one result.
        """,
    )
    legend_item: Optional[Any] = Field(
        default=None,
        description="Some CDR endpoints can allow a legend item data attached to each feature.",
    )
