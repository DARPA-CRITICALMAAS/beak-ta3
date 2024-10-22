from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cdr_schemas.common import (
    CRITICALMAAS_PIXEL,
    GeoJsonType,
    GeomType,
    ModelProvenance,
)


class Point(BaseModel):
    """
    Individual occurance of a point feature.
    """

    coordinates: List[Union[float, int]] = Field(
        description="""The coordinates of the point. Format is expected to be an
                    [x,y] coordinate where the top left is the origin (0,0)."""
    )
    type: GeomType = GeomType.Point


class PointProperties(BaseModel):
    """
    Properties of the Point.
    """

    # Model Provenance
    model: str = Field(description="Name of the model used to generate this data")
    model_version: str = Field(
        description="Version of the model used to generate this data"
    )

    confidence: Optional[Union[float, int]] = Field(
        default=None, description="The prediction confidence of the model"
    )

    # Point Properties
    bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The extacted 2 point bounding box of the point item.
                    Format is expected to be [x1,y1,x2,y2] where the top left
                    is the origin (0,0).""",
    )
    dip: Optional[int] = Field(  # TODO add description
        default=None, description="TODO : Add description"
    )
    dip_direction: Optional[int] = Field(  # TODO add description
        default=None, description="TODO : Add description"
    )
    reference_id: str = Field(
        default="",
        description="""
            Identifies the original CDR item ID from which this new item was derived,
            aiding in tracking provenance.
        """,
    )
    validated: Optional[bool] = Field(None, description="Validated by human")

    model_config = ConfigDict(protected_namespaces=())


class PointFeature(BaseModel):
    """
    Point feature.
    """

    type: GeoJsonType = GeoJsonType.Feature
    id: str = Field(
        description="""Each point geometry has a unique id.
                    The ids are used to link the point geometries is px-coord and geo-coord."""
    )
    geometry: Point
    properties: PointProperties


class PointFeatureCollection(BaseModel):
    """
    All point features for legend item.
    """

    type: GeoJsonType = GeoJsonType.FeatureCollection
    features: List[PointFeature] = Field(
        default_factory=list,
        description="""
            List of point features
        """,
    )


class PointLegendAndFeaturesResult(BaseModel):
    """
    Point legend item metadata along with associated point features found.
    """

    id: str = Field(description="your internal id")

    # Legend Fields
    # TODO move to a more sensible location
    legend_provenance: Optional[ModelProvenance] = Field(
        default=None, description="Where the data originated from."
    )
    name: str = Field(description="Label of the map unit in the legend")
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
    point_features: Optional[PointFeatureCollection] = Field(
        default=None, description="All point features for legend item."
    )
