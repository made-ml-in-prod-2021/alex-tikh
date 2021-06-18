from pydantic import BaseModel


class HeartResponse(BaseModel):
    id: str
    target: int
