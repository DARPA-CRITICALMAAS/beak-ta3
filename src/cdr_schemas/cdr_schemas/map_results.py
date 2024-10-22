from typing import List

from pydantic import BaseModel, Field

from cdr_schemas.feature_results import FeatureResults
from cdr_schemas.georeference import GeoreferenceResults


class MapResults(BaseModel):
    """
    All results for map.
    """

    cog_id: str = Field(
        ...,
        description="""
            Cog id.
        """,
    )
    georef_results: List[GeoreferenceResults] = Field(
        default_factory=list,
        description="""
            A list of georef results from systems.
        """,
    )
    extraction_results: List[FeatureResults] = Field(
        default_factory=list,
        description="""
            A list of feature extraction results from systems.
        """,
    )
