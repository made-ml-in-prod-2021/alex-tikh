from pydantic import BaseModel, conlist
from typing import List, Union


class HeartModel(BaseModel):
    data: List[conlist(Union[int, float, str, None], min_items=13, max_items=13)]
    features: List[str]

