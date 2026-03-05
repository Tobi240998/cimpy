from __future__ import annotations

import json
from typing import List, Optional, Literal, Set, Dict, Any, Callable

from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama
from cimpy_time_analysis.langchain_llm import get_llm



# =============================================================================
# CIM Schema
# =============================================================================

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


def normalize(parsed: QueryParse) -> QueryParse:
    uniq = []
    seen = set()

    for t in parsed.detected_types:
        if t in CIM_TYPES_ALLOWED and t not in seen:
            uniq.append(t)
            seen.add(t)

    return QueryParse(detected_types=uniq, metric=parsed.metric)


# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = f"""
Du interpretierst kurze User-Queries für CIM-Analysen.

Extrahiere folgende Struktur als JSON:

{{
  "detected_types": ["PowerTransformer" | "ConformLoad" | "SvVoltage" | "SvPowerFlow"],
  "metric": "P" | "Q" | "S" | null
}}

Whitelist für detected_types:
{sorted(CIM_TYPES_ALLOWED)}

Synonyme:
Trafo / Transformator -> PowerTransformer
Verbraucher / Last -> ConformLoad
Spannung -> SvVoltage
Leistung / Power -> SvPowerFlow
Wirkleistung -> P
Blindleistung -> Q
Scheinleistung -> S

Regeln:
Nur JSON zurückgeben.
Keinen zusätzlichen Text.
Wenn unklar: leere Liste und metric null.
"""


# =============================================================================
# User interaction
# =============================================================================

def default_ask_user(question: str) -> str:
    return input(f"{question}\n> ").strip()


# =============================================================================
# JSON extraction helper
# =============================================================================

def extract_json(text: str):

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    raise ValueError("No JSON found")


# =============================================================================
# LLM parsing
# =============================================================================

def parse_with_llm(llm, context: str):

    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("user", context)
    ])

    text = getattr(response, "content", str(response))

    data = extract_json(text)

    parsed = QueryParse(**data)

    return normalize(parsed)


# =============================================================================
# Main interpreter
# =============================================================================

def interpret_user_query(
        user_input: str,
        ask_user: Callable[[str], str] | None = None,
        max_rounds: int = 5
) -> Dict[str, Any]:

    if ask_user is None:
        ask_user = default_ask_user

    llm = get_llm()

    context_lines = [f"User: {user_input}"]

    for _ in range(max_rounds):

        context = "\n".join(context_lines)

        try:

            parsed = parse_with_llm(llm, context)

            if parsed.detected_types or parsed.metric:
                return parsed.model_dump()

        except (ValidationError, ValueError) as e:

            repair_prompt = f"""
Der folgende Output konnte nicht geparst werden.

Fehler:
{str(e)}

Kontext:
{context}

Gib die korrekte JSON-Struktur zurück.
"""

            response = llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("user", repair_prompt)
            ])

            try:
                data = extract_json(response.content)
                parsed = QueryParse(**data)
                parsed = normalize(parsed)

                if parsed.detected_types or parsed.metric:
                    return parsed.model_dump()

            except Exception:
                pass

        clarify_prompt = f"""
Die Anfrage ist unklar.

Kontext:
{context}

Stelle eine kurze Rückfrage, um Objekt-Typ oder Metrik zu klären.
Nur eine Frage.
"""

        q = llm.invoke([("user", clarify_prompt)]).content.strip()

        answer = ask_user(q)

        context_lines.append(f"Assistant: {q}")
        context_lines.append(f"User: {answer}")

    return {"detected_types": [], "metric": None}

