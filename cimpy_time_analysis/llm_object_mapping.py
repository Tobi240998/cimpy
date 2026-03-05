from __future__ import annotations

import json
from typing import List, Optional, Literal, Set, Dict, Any, Callable

from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama
from cimpy_time_analysis.langchain_llm import get_llm


# =============================================================================
# 2) CIM Typen / Schema
# =============================================================================

EQUIPMENT_TYPES_ALLOWED = {
    "PowerTransformer",
    "ConformLoad",
}

STATE_TYPES_ALLOWED = {
    "SvVoltage",
    "SvPowerFlow",
}

ALL_TYPES_ALLOWED = EQUIPMENT_TYPES_ALLOWED | STATE_TYPES_ALLOWED

Metric = Optional[Literal["P", "Q", "S"]]


class QueryParse(BaseModel):
    # Neu: getrennt
    equipment_types: List[str] = Field(default_factory=list)
    state_types: List[str] = Field(default_factory=list)

    # optional
    metric: Metric = None


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def normalize(parsed: QueryParse) -> QueryParse:
    # whitelist + dedup
    eq = [t for t in parsed.equipment_types if t in EQUIPMENT_TYPES_ALLOWED]
    st = [t for t in parsed.state_types if t in STATE_TYPES_ALLOWED]

    return QueryParse(
        equipment_types=_dedup_keep_order(eq),
        state_types=_dedup_keep_order(st),
        metric=parsed.metric,
    )


# =============================================================================
# 3) Prompts
# =============================================================================

SYSTEM_PROMPT = f"""
Du interpretierst kurze User-Queries für CIM-Analysen.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{{
  "equipment_types": ["PowerTransformer" | "ConformLoad", ...],
  "state_types": ["SvVoltage" | "SvPowerFlow", ...],
  "metric": "P" | "Q" | "S" | null
}}

Regeln:
- equipment_types darf nur aus {sorted(EQUIPMENT_TYPES_ALLOWED)} bestehen.
- state_types darf nur aus {sorted(STATE_TYPES_ALLOWED)} bestehen.
- metric:
  - P = Wirkleistung (active)
  - Q = Blindleistung (reactive)
  - S = Scheinleistung (apparent)
- Wenn etwas unklar ist: entsprechende Liste leer lassen bzw. metric null.
- Equipment und State sind getrennte Konzepte: gib sie getrennt aus.

Synonyme:
- Trafo/Transformator/Transformer => PowerTransformer
- Verbraucher/Last/Load/ConformLoad => ConformLoad
- Spannung/Voltage => SvVoltage
- Leistung/Power/Powerflow => SvPowerFlow
- Wirkleistung/P => metric "P"
- Blindleistung/Q => metric "Q"
- Scheinleistung/S => metric "S"
""".strip()


# Rückfrage: wir wollen gezielt fehlende Teile klären (Equipment/State/Metric)
CLARIFY_SYSTEM = "Du stellst genau EINE kurze Rückfrage auf Deutsch. Kein Zusatztext."


def make_clarify_prompt(context: str, missing: str) -> List[tuple]:
    """
    missing: "equipment" | "state" | "metric" | "general"
    """
    if missing == "equipment":
        goal = "Kläre, welches Equipment gemeint ist (PowerTransformer oder ConformLoad)."
    elif missing == "state":
        goal = "Kläre, welche StateVariable gemeint ist (SvVoltage oder SvPowerFlow)."
    elif missing == "metric":
        goal = "Kläre welche Metrik gemeint ist: P (Wirkleistung), Q (Blindleistung) oder S (Scheinleistung)."
    else:
        goal = "Kläre die Anfrage so, dass Equipment (PowerTransformer/ConformLoad) und State (SvVoltage/SvPowerFlow) bestimmt werden können."

    user = f"""
Die Anfrage ist unklar oder unvollständig.

Kontext:
{context}

Ziel:
{goal}

Stelle genau EINE kurze Frage.
""".strip()

    return [("system", CLARIFY_SYSTEM), ("user", user)]


# =============================================================================
# 4) User interaction
# =============================================================================

def default_ask_user(question: str) -> str:
    # CLI default; im UI ersetzen durch Callback
    return input(f"{question}\n> ").strip()


# =============================================================================
# 5) JSON extraction helper
# =============================================================================

def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # direkt
    try:
        return json.loads(text)
    except Exception:
        pass

    # wenn Text drumrum -> ersten {...} Block extrahieren
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        return json.loads(snippet)

    raise ValueError("No JSON object found in LLM output.")


# =============================================================================
# 6) LLM parse + repair helper
# =============================================================================

def parse_with_llm(llm, context: str) -> QueryParse:
    resp = llm.invoke([("system", SYSTEM_PROMPT), ("user", context)])
    text = getattr(resp, "content", str(resp))
    data = extract_json(text)
    parsed = QueryParse(**data)
    return normalize(parsed)


def repair_with_llm(llm, context: str, error: Exception, bad_output: str | None = None) -> Optional[QueryParse]:
    repair_prompt = f"""
Dein vorheriger Output konnte nicht in das Schema geparst/validiert werden.

Fehler:
{str(error)}

Kontext:
{context}

Gib jetzt NUR gültiges JSON im geforderten Schema zurück.
""".strip()

    # bad_output optional mitgeben
    if bad_output:
        repair_prompt += f"\n\nVorheriger Output:\n{bad_output}\n"

    resp = llm.invoke([("system", SYSTEM_PROMPT), ("user", repair_prompt)])
    text = getattr(resp, "content", str(resp))
    data = extract_json(text)
    parsed = QueryParse(**data)
    return normalize(parsed)


# =============================================================================
# 7) interpret_user_query: liefert IMMER getrennte Listen (equipment/state)
#    und fragt nach, bis beides vorhanden ist (oder max_rounds)
# =============================================================================

def interpret_user_query(
    user_input: str,
    *,
    ask_user: Optional[Callable[[str], str]] = None,
    max_rounds: int = 6,
    default_state_if_equipment_only: str = "SvPowerFlow",  # dein bisheriges Default-Verhalten vorgezogen
) -> Dict[str, Any]:
    """
    Liefert IMMER ein dict:
      {
        "equipment_types": [...],
        "state_types": [...],
        "metric": "P|Q|S"|None
      }

    Policy:
    - Wir versuchen, Equipment UND State zu bekommen (beides erforderlich).
    - Wenn nur Equipment erkannt ist und kein State: default_state_if_equipment_only setzen (z.B. SvPowerFlow).
      (Damit entspricht es deinem bisherigen "Default Leistungsabfrage".)
    - Wenn weiterhin unklar/leer: Rückfrage-Schleife (mehrfach möglich).
    """
    if ask_user is None:
        ask_user = default_ask_user

    llm = get_llm()

    context_lines: List[str] = [f"User: {user_input}"]

    for _ in range(max_rounds):
        context = "\n".join(context_lines)

        parsed: Optional[QueryParse] = None

        # 1) Normal parse
        try:
            parsed = parse_with_llm(llm, context)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            # 2) Repair parse
            try:
                parsed = repair_with_llm(llm, context, e)
            except Exception:
                parsed = None

        if parsed is None:
            # Rückfrage allgemein
            q = llm.invoke(make_clarify_prompt(context, "general")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        # --- Default-Regel vorziehen (wie in handle_user_query) ---
        # Wenn Equipment erkannt wurde, aber keine StateVariable: default setzen
        if parsed.equipment_types and not parsed.state_types:
            parsed = QueryParse(
                equipment_types=parsed.equipment_types,
                state_types=[default_state_if_equipment_only],
                metric=parsed.metric,
            )
            parsed = normalize(parsed)

        # Wenn wir jetzt beides haben -> fertig
        if parsed.equipment_types and parsed.state_types:
            return parsed.model_dump()

        # Sonst gezielt nach dem fehlenden Teil fragen
        if not parsed.equipment_types:
            missing = "equipment"
        elif not parsed.state_types:
            missing = "state"
        else:
            missing = "general"

        q = llm.invoke(make_clarify_prompt(context, missing)).content.strip()
        a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
        context_lines += [f"Assistant: {q}", f"User: {a}"]

    # Letztes Mittel: gültiges Format zurückgeben
    return {"equipment_types": [], "state_types": [], "metric": None}





