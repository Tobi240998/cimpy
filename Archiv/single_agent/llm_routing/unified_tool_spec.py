from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class UnifiedToolSpec:
    full_name: str
    name: str
    domain: str
    description: str
    input_schema: Dict[str, Any]
    output_schema_hint: Dict[str, Any]
    capability_tags: List[str] = field(default_factory=list)
    mutating: bool = False
    requires_state: List[str] = field(default_factory=list)
    produces_state: List[str] = field(default_factory=list)
    is_summary: bool = False
    domain_notes: List[str] = field(default_factory=list)