from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cdr_schemas.common import (
    CRITICALMAAS_PIXEL,
    GeoJsonType,
    GeomType,
    ModelProvenance,
)


class DashType(str, Enum):
    none = ""
    solid = "solid"
    dash = "dash"
    dotted = "dotted"


class Line(BaseModel):
    """
    Individual line segmentation of a line feature.
    """

    coordinates: List[List[Union[float, int]]] = Field(
        description="""The coordinates of the line. Format is expected to
                    be [x,y] coordinate pairs where the top left is the origin
                    (0,0)."""
    )
    type: GeomType = GeomType.LineString


class LineProperties(BaseModel):
    """
    Properties of the line.
    """

    # Model Provenance
    model: str = Field(description="Name of the model used to generate this data")
    model_version: str = Field(
        description="Version of the model used to generate this data"
    )
    confidence: Optional[Union[float, int]] = Field(
        default=None, description="The prediction confidence of the model"
    )
    # Line Properties
    dash_pattern: DashType = Field(
        default=DashType.none, description="values = {solid, dash, dotted}"
    )
    symbol: str = Field(default="", description="TODO : Add description")
    reference_id: str = Field(
        default="",
        description="""
            Identifies the original CDR item ID from which this new item was derived,
            aiding in tracking provenance.
        """,
    )
    validated: Optional[bool] = Field(None, description="Validated by human")
    model_config = ConfigDict(protected_namespaces=())


class LineFeature(BaseModel):
    """
    Line Feature.
    """

    type: GeoJsonType = GeoJsonType.Feature
    id: str = Field(
        description="""Each line geometry has a unique id.
                    The ids are used to link the line geometries is px-coord and geo-coord."""
    )
    geometry: Line
    properties: LineProperties


class LineFeatureCollection(BaseModel):
    """
    All line features for legend item.
    """

    type: GeoJsonType = GeoJsonType.FeatureCollection
    features: List[LineFeature] = Field(
        default_factory=list,
        description="""
            List of all line features
        """,
    )


class LineLegendAndFeaturesResult(BaseModel):
    """
    Line legend item with metadata and associated line features found.
    """

    id: str = Field(description="your internal id")

    # Legend Fields
    # TODO move to a more sensible location
    legend_provenance: Optional[ModelProvenance] = Field(
        default=None, description="Where the data originated from."
    )
    name: str = Field(default="", description="Label of the map unit in the legend")
    abbreviation: str = Field(
        default="", description="Abbreviation of the map unit label."
    )
    description: str = Field(
        default="", description="Description of the map unit in the legend"
    )
    legend_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the map units label.
                    Format is expected to be [x1,y1,x2,y2] where the top left
                    is the origin (0,0).""",
    )
    legend_contour: List[List[Union[float, int]]] = Field(
        default_factory=list,
        description="""The more precise polygon bounding box of the map units
                    label. Format is expected to be [x,y] coordinate pairs
                    where the top left is the origin (0,0).""",
    )
    reference_id: str = Field(
        default="",
        description="""
            Identifies the original CDR item ID from which this new item was derived,
            aiding in tracking provenance.
        """,
    )
    validated: Optional[bool] = Field(None, description="Validated by human")

    # Segmentation Fields
    crs: str = Field(
        default=CRITICALMAAS_PIXEL,
        description="""What projection the geometry of the segmentation are in,
                    Default is CRITICALMAAS_PIXEL which specifies pixel coordinates.
                    Possible values are {CRITICALMAAS_PIXEL, EPSG:*}""",
    )
    cdr_projection_id: str = Field(
        default="",
        description="""If non-pixel coordinates are used the cdr projection id of the
                    georeference that was used to create them is required.""",
    )
    line_features: Optional[LineFeatureCollection] = None
