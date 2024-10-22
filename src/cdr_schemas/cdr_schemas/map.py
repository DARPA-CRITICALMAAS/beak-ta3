from pydantic import BaseModel, Field


class MapProvenance(BaseModel):
    """JSON model for Document Provenance"""

    system_name: str = Field(..., description="Name of system storing map")
    id: str = Field(None, description="The system ID of the map")
    url: str = Field(None, description="URL of map at system storing map")


class Map(BaseModel):
    """JSON model for 'Map''"""

    id: str = Field(..., description="The CDR ID of the Map")
    provenance: list[MapProvenance] = Field(
        default_factory=list, description="provenance list"
    )
    is_open: bool = Field(
        ...,
        description="Whether map is open or not.",
    )

    system: str = Field(
        ...,
        description="""
            The name of the system used.
        """,
    )
    system_version: str = Field(
        ...,
        description="""
            The version of the system used.
        """,
    )
