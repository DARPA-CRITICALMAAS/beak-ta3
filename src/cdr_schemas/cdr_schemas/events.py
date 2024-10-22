from typing import Any

from pydantic import BaseModel


class Event(BaseModel):
    id: str
    event: str
    payload: Any | None


class MapEventPayload(BaseModel):
    cog_id: str
    cog_url: str
