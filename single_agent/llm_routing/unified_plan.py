from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UnifiedPlanStep(BaseModel):
    domain: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    save_as: Optional[str] = None
    description: str = ""
    mutating: bool = False


class UnifiedPlan(BaseModel):
    domain: str
    user_input: str
    steps: List[UnifiedPlanStep]
    classification: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""