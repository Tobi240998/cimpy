from pydantic import BaseModel, Field
from typing import List, Literal

PowerFactoryResultMetric = Literal[
    "bus_voltage",
    "bus_p",
    "bus_q",
    "line_loading",
]


class LoadChangeInstruction(BaseModel):
    action: Literal["change_load"] = "change_load"
    load_name: str
    delta_p_mw: float
    result_requests: List[PowerFactoryResultMetric] = Field(
        default_factory=lambda: ["bus_voltage"],
        description=(
            "Selektiv angeforderte Ergebnis-Metriken für den Vorher/Nachher-Vergleich. "
            "Standardmäßig werden Bus-Spannungen verglichen."
        ),
    )
