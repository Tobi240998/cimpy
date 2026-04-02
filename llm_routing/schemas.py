from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# Erzeugung von Struktur der Antworten

class HistoricalArgs(BaseModel):
    user_input: str = Field(..., description="Originale Nutzeranfrage oder normalisierte Frage")
    equipment_type: Optional[str] = Field(None, description="z.B. transformer/line/busbar/...")
    equipment_id: Optional[str] = Field(None, description="Konkrete ID falls bekannt")
    equipment_name: Optional[str] = Field(None, description="Name/Label falls bekannt")
    time_range: Optional[str] = Field(None, description="z.B. '2026-03-02', 'gestern', '2026-03-01..2026-03-02'")
    metric: Optional[str] = Field(None, description="z.B. loading_pct, p_mw, u_kv")
    aggregation: Optional[str] = Field(None, description="avg|min|max|timeseries")


class PowerFactoryArgs(BaseModel):
    user_input: str = Field(..., description="Originale Nutzeranfrage")
    project: Optional[str] = None
    change: Optional[str] = None


# --- Router Actions (Structured Output) ---

class CallToolAction(BaseModel):
    action: Literal["call_tool"]
    tool: Literal["historical", "powerfactory"]
    args: Dict[str, Any] = Field(default_factory=dict)


class AskUserAction(BaseModel):
    action: Literal["ask_user"]
    question: str
    missing_fields: List[str] = Field(default_factory=list)
    partial: Dict[str, Any] = Field(default_factory=dict)
    intended_tool: Optional[Literal["historical", "powerfactory"]] = None


RouterAction = Union[CallToolAction, AskUserAction]