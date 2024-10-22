from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetaData(BaseModel):
    doi: str = Field("", description="Document DOI")
    authors: List[str] = Field(description="Document Authors", default_factory=list)
    journal: str = Field("", description="Document Journal")
    year: Optional[int] = Field(None, description="year")
    month: Optional[int] = Field(None, description="month")
    volume: Optional[int] = Field(None, description="volume")
    issue: str = Field("", description="issue")
    description: str = Field("", description="description")
    publisher: str = Field("", description="publisher")


class DocumentProvenance(BaseModel):
    """JSON model for Document Provenance"""

    external_system_name: str = Field(
        ..., description="Name of system storing document"
    )
    external_system_id: str = Field("", description="The system ID of the document")
    external_system_url: str = Field("", description="Name of system storing document")


class UploadDocument(BaseModel):
    """JSON model for uploading new document"""

    title: str = Field(..., description="Title of the document")
    is_open: bool = Field(
        True,
        description="Whether document is open or not.",
    )

    provenance: list[DocumentProvenance] = Field(
        description="provenance list", default_factory=list
    )
    metadata: Optional[DocumentMetaData] = Field(None, description="document metadata")

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


class Document(BaseModel):
    """JSON model for user-facing document metadata"""

    id: str = Field(..., description="The internal ID of the document")
    title: str = Field(..., description="Title of the document")
    is_open: bool = Field(
        ...,
        description="Whether document is open or not.",
    )

    pages: int = Field(..., description="Document page count")
    size: int = Field(..., description="Document size in bytes")

    provenance: list[DocumentProvenance] = Field(
        ..., description="provenance list", default_factory=list
    )
    metadata: Optional[DocumentMetaData] = Field(None, description="document metadata")

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


class DocumentExtraction(BaseModel):
    """JSON model for user-facing document metadata"""

    id: str | None = Field(None, description="The internal ID of the xtraction")
    document_id: str = Field(None, description="The internal ID of the source document")
    extraction_type: str = Field(
        ..., description="The type of model that produced the extraction"
    )
    extraction_label: str = Field(
        ..., description="The classification of the extraction within its model"
    )
    score: float | None = Field(None, description="The confidence of the extraction")
    bbox: tuple[float, float, float, float] | None = Field(
        None, description="The bounding box of the extraction"
    )
    page_num: int | None = Field(None, description="The page number of the extraction")
    external_link: str | None = Field(None, description="A link to the extraction")
    data: Dict | None = Field(
        None, description="Extra information about the extraction"
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
