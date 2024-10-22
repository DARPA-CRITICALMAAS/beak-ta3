from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cdr_schemas.common import GeomType


class AreaType(str, Enum):
    Map_Area = "map_area"
    Legend_Area = "legend_area"
    CrossSection = "cross_section"
    OCR = "ocr"
    Polygon_Legend_Area = "polygon_legend_area"
    Line_Point_Legend_Area = "line_point_legend_area"
    Line_Legend_Area = "line_legend_area"
    Point_Legend_Area = "point_legend_area"
    Correlation_Diagram = "correlation_diagram"


class Area_Extraction(BaseModel):
    """
    Area extraction of a cog.
    """

    type: GeomType = GeomType.Polygon
    coordinates: List[List[List[Union[float, int]]]] = Field(
        description="""The coordinates of the areas boundry. Format is expected
                    to be [x,y] coordinate pairs where the top left is the
                    origin (0,0)."""
    )
    bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The extracted bounding box of the area.
                    Format is expected to be [x1,y1,x2,y2] where the top left
                    is the origin (0,0).""",
    )
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
    reference_id: str = Field(
        default="",
        description="""
            Identifies the original CDR item ID from which this new item was derived,
            aiding in tracking provenance.
        """,
    )
    validated: bool = Field(False, description="Validated by human")

    # Model Provenance
    model: str = Field(description="Name of the model used to generate this data")
    model_version: str = Field(
        description="Version of the model used to generate this data"
    )
    confidence: Optional[Union[float, int]] = Field(
        default=None, description="The prediction confidence of the model"
    )

    model_config = ConfigDict(protected_namespaces=())
