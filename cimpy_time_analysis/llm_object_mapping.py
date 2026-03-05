from __future__ import annotations

import json
from typing import List, Optional, Literal, Set

from pydantic import BaseModel, Field, ValidationError

from langchain_ollama import ChatOllama
from cimpy_time_analysis.langchain_llm import get_llm


# --- Ziel-Schema (was interpret_user_query liefern soll) ----------------------

CIM_TYPES_ALLOWED = {
    "PowerTransformer",
    "ConformLoad",
    "SvVoltage",
    "SvPowerFlow",
}

Metric = Optional[Literal["P", "Q", "S"]]


class QueryParse(BaseModel):
    detected_types: List[str] = Field(default_factory=list)
    metric: Metric = None

    # Optional: härter validieren, damit nix “halluziniert”
    def normalized(self) -> "QueryParse":
        # unique + filter only allowed
        unique: List[str] = []
        seen: Set[str] = set()
        for t in self.detected_types:
            if t in CIM_TYPES_ALLOWED and t not in seen:
                unique.append(t)
                seen.add(t)
        return QueryParse(detected_types=unique, metric=self.metric)


# --- LLM-basierte Interpretation --------------------------------------------

SYSTEM_INSTRUCTIONS = f"""
Du interpretierst kurze User-Queries für CIM-Analysen.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Erklärtext.

JSON-Schema:
{{
  "detected_types": [<string>, ...],
  "metric": "P" | "Q" | "S" | null
}}

Regeln:
- detected_types darf nur Werte aus dieser Whitelist enthalten: {sorted(CIM_TYPES_ALLOWED)}
- metric bedeutet:
  - "P" = Wirkleistung / active power
  - "Q" = Blindleistung / reactive power
  - "S" = Scheinleistung / apparent power
- Wenn nichts eindeutig ist: detected_types leere Liste und metric null.

Synonyme/Hinweise:
- Trafo/Transformator/Transformer => PowerTransformer
- Verbraucher/Last/Load/ConformLoad => ConformLoad
- Spannung/Voltage => SvVoltage
- Leistung/Power/Powerflow => SvPowerFlow
- Wirkleistung/P => metric "P"
- Blindleistung/Q => metric "Q"
- Scheinleistung/S => metric "S"
""".strip()


def interpret_user_query(user_input: str) -> dict:
    """
    LLM-Version der bisherigen Keyword-Map.
    Gibt zurück: {"detected_types": [...], "metric": "P"/"Q"/"S"/None}
    """

    llm = get_llm()

    # Variante A (bevorzugt): Structured Output, wenn von deiner LangChain-Version unterstützt
    try:
        structured = llm.with_structured_output(QueryParse)  # type: ignore[attr-defined]
        result: QueryParse = structured.invoke(
            [
                ("system", SYSTEM_INSTRUCTIONS),
                ("user", user_input),
            ]
        )
        result = result.normalized()
        return result.model_dump()
    except Exception:
        # Fallback: strikt JSON anfordern und selbst parsen/validieren
        pass

    # Variante B (Fallback): JSON-only + json.loads + Pydantic-Validation
    msg = llm.invoke(
        [
            ("system", SYSTEM_INSTRUCTIONS),
            ("user", user_input),
        ]
    )

    # ChatOllama liefert typischerweise AIMessage mit .content
    content = getattr(msg, "content", msg)
    if not isinstance(content, str):
        content = str(content)

    # Robustness: evtl. führende/trailing Whitespaces
    content = content.strip()

    try:
        data = json.loads(content)
        parsed = QueryParse(**data).normalized()
        return parsed.model_dump()
    except (json.JSONDecodeError, ValidationError):
        # Ultra-konservativ: wenn Parsing/Validation scheitert -> “nichts erkannt”
        return {"detected_types": [], "metric": None}


