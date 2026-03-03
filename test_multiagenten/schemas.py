from pydantic import BaseModel
from typing import Literal

class LoadChangeInstruction(BaseModel):
    action: Literal["change_load"] = "change_load"
    load_name: str
    delta_p_mw: float

