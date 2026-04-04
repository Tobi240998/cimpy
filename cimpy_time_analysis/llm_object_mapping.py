from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import List, Optional, Literal, Set, Dict, Any, Callable, Tuple

from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from datetime import datetime, timedelta, timezone


# =============================================================================
# 1) LLM
# =============================================================================

def get_llm():
    return ChatOllama(
        model="qwen3:30b",
        base_url="http://localhost:11434",
        temperature=0.0,
        streaming=False,
        timeout=180,
    )


# =============================================================================
# 2) Matching Utilities
# =============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = t.replace("transformator", "trafo")
    t = t.replace("trf", "trafo")
    t = t.replace("/", "-")
    t = re.sub(r"\s+", "", t)
    return t


def extract_two_numbers(text: str):
    if not text:
        return None
    m = re.search(r"(\d+)\D+(\d+)", text)
    if not m:
        return None
    return m.group(1), m.group(2)


def extract_one_number(text: str):
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    return m.group(1)


def _number_boundary_match(num: str, norm_name: str) -> bool:
    if not num or not norm_name:
        return False
    pattern = rf"(^|[^0-9]){re.escape(num)}([^0-9]|$)"
    return re.search(pattern, norm_name) is not None


def equipment_identifier(eq: Any) -> Optional[str]:
    for attr in ("mRID", "mrid", "rdfId", "rdfid", "id", "uuid", "UID", "uid"):
        v = getattr(eq, attr, None)
        if v:
            return str(v)
    try:
        if isinstance(eq, dict):
            for k in ("mRID", "mrid", "rdfId", "rdfid", "id", "uuid", "uid"):
                if k in eq and eq[k]:
                    return str(eq[k])
    except Exception:
        pass
    return None


# =============================================================================
# 3) Defaults / Schema
# =============================================================================

DEFAULT_EQUIPMENT_TYPES = {
    "PowerTransformer",
    "ConformLoad",
}

DEFAULT_STATE_TYPES = {
    "SvVoltage",
    "SvPowerFlow",
}

POWERFLOW_TARGET_EQUIPMENT_TYPES = {
    "SynchronousMachine",
    "AsynchronousMachine",
    "ConformLoad",
    "EnergyConsumer",
    "PowerTransformer",
    "ACLineSegment",
    "EquivalentInjection",
    "ExternalNetworkInjection",
}

VOLTAGE_TARGET_EQUIPMENT_TYPES = {
    "BusbarSection",
    "SynchronousMachine",
    "AsynchronousMachine",
    "ConformLoad",
    "EnergyConsumer",
    "PowerTransformer",
    "ACLineSegment",
    "EquivalentInjection",
    "ExternalNetworkInjection",
}

Metric = Optional[Literal["P", "Q", "S"]]


BASE_ATTRIBUTE_SPECS: Dict[str, Dict[str, Any]] = {
    "name": {"aliases": ["name", "bezeichnung", "gerätebezeichnung", "equipment name"]},
    "mRID": {"aliases": ["mrid", "m rid", "id", "uuid", "kennung", "technical id"]},
    "description": {"aliases": ["description", "beschreibung", "desc"]},
    "ratedS": {"aliases": ["rateds", "rated s", "rated power", "bemessungsleistung", "nennleistung", "rating"]},
    "ratedU": {"aliases": ["ratedu", "rated u", "rated voltage", "bemessungsspannung", "nennspannung"]},
    "p": {"aliases": ["p", "wirkleistung", "active power", "real power"]},
    "q": {"aliases": ["q", "blindleistung", "reactive power"]},
    "maxQ": {"aliases": ["maxq", "max q", "max reactive power", "blindleistungsobergrenze", "maximale blindleistung"]},
    "minQ": {"aliases": ["minq", "min q", "min reactive power", "blindleistungsuntergrenze", "minimale blindleistung"]},
    "initialP": {"aliases": ["initialp", "initial p", "initial active power", "startwert wirkleistung", "initialleistung"]},
    "nominalP": {"aliases": ["nominalp", "nominal p", "nominal active power", "nennwirkleistung", "nominalleistung"]},
    "maxOperatingP": {"aliases": ["maxoperatingp", "max operating p", "max operating power", "maximale wirkleistung", "maximalleistung"]},
    "minOperatingP": {"aliases": ["minoperatingp", "min operating p", "min operating power", "minimale wirkleistung", "minimalleistung"]},
    "operatingMode": {"aliases": ["operatingmode", "operating mode", "betriebsmodus", "betriebspunktmodus"]},
    "type": {"aliases": ["type", "typ", "equipment type", "gerätetyp"]},
}



class EquipmentSelection(BaseModel):
    equipment_type: str
    equipment_key: str
    equipment_name: Optional[str] = None
    equipment_id: Optional[str] = None


class QueryParse(BaseModel):
    equipment_detected: List[str] = Field(default_factory=list)
    state_detected: List[str] = Field(default_factory=list)
    metric: Metric = None
    equipment_selection: List[EquipmentSelection] = Field(default_factory=list)
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    time_label: Optional[str] = None


class CandidateChoice(BaseModel):
    equipment_key: Optional[str] = None
    need_clarification: bool = False
    clarification_question: Optional[str] = None

class BaseAttributeSelectionDecision(BaseModel):
    selected_attributes: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(default="")


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _normalize_allowed_set(values, fallback: Set[str]) -> Set[str]:
    if not values:
        return set(fallback)

    out = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.add(s)
    return out if out else set(fallback)


def normalize_query(
    parsed: QueryParse,
    *,
    allowed_equipment_types: Optional[Set[str]] = None,
    allowed_state_types: Optional[Set[str]] = None,
) -> QueryParse:
    allowed_equipment_types = _normalize_allowed_set(allowed_equipment_types, DEFAULT_EQUIPMENT_TYPES)
    allowed_state_types = _normalize_allowed_set(allowed_state_types, DEFAULT_STATE_TYPES)

    eq = [t for t in parsed.equipment_detected if t in allowed_equipment_types]
    st = [t for t in parsed.state_detected if t in allowed_state_types]

    filtered_selection = [
        sel for sel in (parsed.equipment_selection or [])
        if sel.equipment_type in allowed_equipment_types
    ]

    return QueryParse(
        equipment_detected=_dedup_keep_order(eq),
        state_detected=_dedup_keep_order(st),
        metric=parsed.metric,
        equipment_selection=filtered_selection,
        time_start=parsed.time_start,
        time_end=parsed.time_end,
        time_label=parsed.time_label,
    )


# =============================================================================
# 4) Prompt Builders
# =============================================================================

def build_system_prompt_parse(
    allowed_equipment_types: Set[str],
    allowed_state_types: Set[str],
) -> str:
    allowed_equipment_types_sorted = sorted(allowed_equipment_types)
    allowed_state_types_sorted = sorted(allowed_state_types)

    equipment_type_hint_lines = "\n".join(f"- {t}" for t in allowed_equipment_types_sorted)
    state_type_hint_lines = "\n".join(f"- {t}" for t in allowed_state_types_sorted)

    return f"""
Du interpretierst kurze User-Queries für CIM-Analysen.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{{
  "equipment_detected": ["<equipment type>", ...],
  "state_detected": ["<state type>", ...],
  "metric": "P" | "Q" | "S" | null,
  "time_start": "ISO-8601 datetime in UTC" | null,
  "time_end": "ISO-8601 datetime in UTC" | null,
  "time_label": "kurze Beschreibung des Zeitraums" | null
}}

Erlaubte equipment_detected-Werte:
{equipment_type_hint_lines}

Erlaubte state_detected-Werte:
{state_type_hint_lines}

Regeln:
- equipment_detected darf nur aus {allowed_equipment_types_sorted} bestehen.
- state_detected darf nur aus {allowed_state_types_sorted} bestehen.
- metric:
  - P = Wirkleistung
  - Q = Blindleistung
  - S = Scheinleistung
- Wenn etwas unklar ist: entsprechende Liste leer lassen bzw. Feld auf null setzen.
- time_start und time_end sollen, wenn möglich, als UTC-ISO-Strings ausgegeben werden.
- Für "am 2026-01-09" gilt:
  - time_start = "2026-01-09T00:00:00+00:00"
  - time_end   = "2026-01-10T00:00:00+00:00"
- Für "09.01.2026" gilt dasselbe Datum wie für "2026-01-09".
- Für "zwischen 09.01.2026 und 11.01.2026" gilt:
  - start = Beginn des ersten Tages
  - end   = Beginn des Tages NACH dem letzten Datum
- Für "vom 09.01.2026 bis 11.01.2026" gilt dasselbe.
- Für "gestern" / "heute" darfst du relative Zeiträume interpretieren.
- time_label soll kurz sein.

Synonyme:
- Trafo/Transformator/Transformer => PowerTransformer
- Verbraucher/Last/Load/ConformLoad => ConformLoad
- Spannung/Voltage => SvVoltage
- Leistung/Power/Powerflow => SvPowerFlow
- Allgemeine Leistung/Power/power output => meist metric "P"
- Wirkleistung/active power/real power/P => metric "P"
- Blindleistung/reactive power/Q => metric "Q"
- Scheinleistung/apparent power/S => metric "S"
- Loading/Utilization/Auslastung => meist metric "S"
- Auslastung eines Trafos => meist metric "S"
""".strip()


SYSTEM_PROMPT_SELECT = """
Du wählst aus einer Kandidatenliste genau EIN Element aus.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{
  "equipment_key": "<genau einer der candidate_keys>" | null,
  "need_clarification": true | false,
  "clarification_question": "<kurze Frage>" | null
}

Regeln:
1) Wenn der Kontext das Equipment klar genug bestimmt, gib direkt equipment_key zurück.
2) Stelle KEINE unnötige Bestätigungsfrage.
3) Wenn der Nutzer im Verlauf bereits etwas Spezifisches wie "Load 27", "Last 27" oder "Trafo 19-20" genannt hat
   und das klar zu einem Kandidaten passt, wähle direkt diesen Kandidaten.
4) Nur wenn es wirklich noch mehrdeutig ist:
   - equipment_key = null
   - need_clarification = true
   - kurze Rückfrage
5) equipment_key muss exakt einem candidate_key entsprechen, wenn gesetzt.
""".strip()


CLARIFY_SYSTEM = "Du stellst genau EINE kurze Rückfrage auf Deutsch. Kein Zusatztext."


METRIC_INFERENCE_SYSTEM = """
Du interpretierst, welche elektrische Leistungsmetrik in einer User-Anfrage gemeint ist.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{
  "metric": "P" | "Q" | "S" | null
}

Regeln:
- P = Wirkleistung / active power / real power / power output / MW
- Q = Blindleistung / reactive power / MVAr
- S = Scheinleistung / apparent power / loading / utilization / MVA
- Allgemeine Formulierungen wie "Leistung", "power", "Leistungswert", "power output" meinen normalerweise P.
- Wenn der Nutzer explizit Blindleistung oder reactive power meint, gib Q zurück.
- Wenn der Nutzer explizit Scheinleistung, apparent power, loading, utilization oder Auslastung meint, gib S zurück.
- Wenn es wirklich nicht ableitbar ist, gib null zurück.
""".strip()


def infer_metric_from_context(llm, context: str) -> Metric:
    user_prompt = f"""
Kontext:
{context}
""".strip()

    try:
        resp = llm.invoke([("system", METRIC_INFERENCE_SYSTEM), ("user", user_prompt)])
        text = getattr(resp, "content", str(resp))
        data = extract_json(text)
        metric = data.get("metric")
        if metric in {"P", "Q", "S"}:
            return metric
    except Exception:
        pass

    context_l = (context or "").lower()
    if any(w in context_l for w in ["blindleistung", "reactive power", "mvar"]):
        return "Q"
    if any(w in context_l for w in ["scheinleistung", "apparent power", "loading", "utilization", "auslastung", "mva"]):
        return "S"
    if any(w in context_l for w in ["wirkleistung", "active power", "real power", "power output", "leistung", " power", "mw"]):
        return "P"
    return None



POWERFLOW_TYPE_INFERENCE_SYSTEM = """
Du interpretierst für eine Leistungs- oder PowerFlow-Anfrage, welcher konkrete netzseitige CIM-Equipment-Typ verwendet werden soll.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{
  "equipment_type": "<genau einer der erlaubten Typen>" | null
}

Regeln:
- Wähle nur einen Typ aus der erlaubten Liste.
- Für Generator/Erzeuger/generator/mechine im elektrischen Netz ist meist SynchronousMachine der richtige Zieltyp.
- GeneratingUnit ist keine gute Zielwahl für terminalbasierte SvPowerFlow-Abfragen, wenn ein netzseitiger Maschinentyp verfügbar ist.
- Für Last/Verbraucher/load ist meist ConformLoad oder EnergyConsumer passend.
- Für Trafo/Transformator/transformer ist PowerTransformer passend.
- Für Leitung/line/kabel ist ACLineSegment passend.
- Wenn kein sicherer Typ aus der erlaubten Liste passt, gib null zurück.
""".strip()


def infer_powerflow_equipment_type_from_context(
    llm,
    context: str,
    *,
    available_equipment_types: List[str],
) -> Optional[str]:
    allowed = [t for t in available_equipment_types if t in POWERFLOW_TARGET_EQUIPMENT_TYPES]
    if not allowed:
        return None

    user_prompt = f"""
Kontext:
{context}

Erlaubte netzseitige PowerFlow-Zieltypen:
{chr(10).join(f"- {t}" for t in allowed)}
""".strip()

    try:
        resp = llm.invoke([("system", POWERFLOW_TYPE_INFERENCE_SYSTEM), ("user", user_prompt)])
        text = getattr(resp, "content", str(resp))
        data = extract_json(text)
        equipment_type = data.get("equipment_type")
        if equipment_type in allowed:
            return equipment_type
    except Exception:
        pass

    context_l = (context or "").lower()
    if any(w in context_l for w in ["generator", "erzeuger", "synchronousmachine", "synchronous machine"]):
        if "SynchronousMachine" in allowed:
            return "SynchronousMachine"
    if any(w in context_l for w in ["asynchronousmachine", "asynchronous machine", "induction machine"]):
        if "AsynchronousMachine" in allowed:
            return "AsynchronousMachine"
    if any(w in context_l for w in ["last", "verbraucher", "load"]):
        if "ConformLoad" in allowed:
            return "ConformLoad"
        if "EnergyConsumer" in allowed:
            return "EnergyConsumer"
    if any(w in context_l for w in ["trafo", "transformator", "transformer"]):
        if "PowerTransformer" in allowed:
            return "PowerTransformer"
    if any(w in context_l for w in ["leitung", "line", "kabel"]):
        if "ACLineSegment" in allowed:
            return "ACLineSegment"

    return None



VOLTAGE_TYPE_INFERENCE_SYSTEM = """
Du interpretierst für eine Spannungs- oder SvVoltage-Anfrage, welcher konkrete netzseitige CIM-Equipment-Typ verwendet werden soll.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{
  "equipment_type": "<genau einer der erlaubten Typen>" | null
}

Regeln:
- Wähle nur einen Typ aus der erlaubten Liste.
- Für Generator/Erzeuger/generator/machine im elektrischen Netz ist meist SynchronousMachine der richtige Zieltyp.
- GeneratingUnit ist keine gute Zielwahl für terminal- bzw. knotengebundene SvVoltage-Abfragen, wenn ein netzseitiger Maschinentyp verfügbar ist.
- Für Bus/Sammelschiene/busbar ist BusbarSection passend.
- Für Last/Verbraucher/load ist meist ConformLoad oder EnergyConsumer passend.
- Für Trafo/Transformator/transformer ist PowerTransformer passend.
- Für Leitung/line/kabel ist ACLineSegment passend.
- Wenn kein sicherer Typ aus der erlaubten Liste passt, gib null zurück.
""".strip()


def infer_voltage_equipment_type_from_context(
    llm,
    context: str,
    *,
    available_equipment_types: List[str],
) -> Optional[str]:
    allowed = [t for t in available_equipment_types if t in VOLTAGE_TARGET_EQUIPMENT_TYPES]
    if not allowed:
        return None

    user_prompt = f"""
Kontext:
{context}

Erlaubte netzseitige Spannungs-Zieltypen:
{chr(10).join(f"- {t}" for t in allowed)}
""".strip()

    try:
        resp = llm.invoke([("system", VOLTAGE_TYPE_INFERENCE_SYSTEM), ("user", user_prompt)])
        text = getattr(resp, "content", str(resp))
        data = extract_json(text)
        equipment_type = data.get("equipment_type")
        if equipment_type in allowed:
            return equipment_type
    except Exception:
        pass

    context_l = (context or "").lower()
    if any(w in context_l for w in ["busbarsection", "busbar section", "busbar", "sammelschiene", "bus "]):
        if "BusbarSection" in allowed:
            return "BusbarSection"
    if any(w in context_l for w in ["generator", "erzeuger", "synchronousmachine", "synchronous machine"]):
        if "SynchronousMachine" in allowed:
            return "SynchronousMachine"
    if any(w in context_l for w in ["asynchronousmachine", "asynchronous machine", "induction machine"]):
        if "AsynchronousMachine" in allowed:
            return "AsynchronousMachine"
    if any(w in context_l for w in ["last", "verbraucher", "load"]):
        if "ConformLoad" in allowed:
            return "ConformLoad"
        if "EnergyConsumer" in allowed:
            return "EnergyConsumer"
    if any(w in context_l for w in ["trafo", "transformator", "transformer"]):
        if "PowerTransformer" in allowed:
            return "PowerTransformer"
    if any(w in context_l for w in ["leitung", "line", "kabel"]):
        if "ACLineSegment" in allowed:
            return "ACLineSegment"

    return None




BASE_ATTRIBUTE_SELECTION_SYSTEM = """
Du interpretierst, welche statischen CIM-Basisattribute in einer User-Anfrage gemeint sind.

Gib AUSSCHLIESSLICH JSON zurück, ohne Markdown, ohne Zusatztext.

Schema:
{
  "selected_attributes": ["<genau einer der erlaubten Attributnamen>", ...],
  "confidence": "high" | "medium" | "low",
  "rationale": "<kurze Begründung>"
}

Regeln:
- Wähle nur Attributnamen aus der erlaubten Liste.
- Erfinde niemals Attributnamen.
- Führe keine Tippfehlerkorrektur, keine semantische Annäherung und kein nearest-match durch.
- Wenn der Nutzer einen technischen Attributnamen nennt, darf nur dann gemappt werden, wenn dieser Name exakt oder nach sehr naheliegender Formatnormalisierung (Leerzeichen, Bindestriche, Groß/Kleinschreibung, Unterstriche) zu einem erlaubten Attribut oder Alias passt.
- Mappe unbekannte technische Namen NICHT auf "ähnliche" Attribute.
- Berücksichtige deutsche und englische Begriffe.
- Wähle nur Attribute, die fachlich wirklich zur Anfrage passen.
- Wenn nichts sicher passt, gib eine leere Liste zurück.
""".strip()


def normalize_attr_text(s: str) -> str:
    return re.sub(r"[\s_\-]", "", (s or "").lower())


def _iter_base_attribute_values(equipment_obj: Any, attr_name: str) -> List[Any]:
    values: List[Any] = []

    if equipment_obj is None or not attr_name:
        return values

    if hasattr(equipment_obj, attr_name):
        value = getattr(equipment_obj, attr_name, None)
        if value is not None:
            values.append(value)

    class_name = equipment_obj.__class__.__name__

    if class_name == "PowerTransformer":
        ends = getattr(equipment_obj, "PowerTransformerEnd", None) or []
        for end in ends:
            if hasattr(end, attr_name):
                value = getattr(end, attr_name, None)
                if value is not None:
                    values.append(value)

    if class_name == "SynchronousMachine":
        generating_unit = getattr(equipment_obj, "GeneratingUnit", None)
        if generating_unit is not None and hasattr(generating_unit, attr_name):
            value = getattr(generating_unit, attr_name, None)
            if value is not None:
                values.append(value)

    return values


def _get_available_base_attributes(equipment_obj: Any) -> List[str]:
    available: List[str] = []
    for attr_name in BASE_ATTRIBUTE_SPECS.keys():
        if _iter_base_attribute_values(equipment_obj, attr_name):
            available.append(attr_name)
    return available


def _llm_match_base_attributes(
    llm,
    user_input: str,
    available_attributes: List[str],
    equipment_obj: Any,
) -> BaseAttributeSelectionDecision:
    parser = PydanticOutputParser(pydantic_object=BaseAttributeSelectionDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", BASE_ATTRIBUTE_SELECTION_SYSTEM.replace("{", "{{").replace("}", "}}") + "\n\n{format_instructions}"),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Equipment class: {equipment_class}\n"
            "Available attributes:\n{available_attributes}\n"
        ),
    ])

    attr_lines = []
    for attr_name in available_attributes:
        values = _iter_base_attribute_values(equipment_obj, attr_name)
        example_value = values[0] if values else None
        aliases = BASE_ATTRIBUTE_SPECS.get(attr_name, {}).get("aliases", []) or []
        alias_text = ", ".join(aliases[:6])
        attr_lines.append(f"- {attr_name} | example_value={example_value!r} | aliases={alias_text}")

    chain = prompt | llm | parser
    decision = chain.invoke({
        "user_input": user_input,
        "equipment_class": equipment_obj.__class__.__name__,
        "available_attributes": "\n".join(attr_lines),
        "format_instructions": parser.get_format_instructions(),
    })
    return decision


def resolve_requested_base_attributes(
    user_input: str,
    equipment_obj: Any,
) -> Dict[str, Any]:
    available_attributes = _get_available_base_attributes(equipment_obj)
    if not available_attributes:
        return {
            "selected_attributes": [],
            "resolution_mode": "no_available_attributes",
            "available_attributes": [],
        }

    llm = get_llm()
    try:
        decision = _llm_match_base_attributes(
            llm=llm,
            user_input=user_input,
            available_attributes=available_attributes,
            equipment_obj=equipment_obj,
        )
        selected = [attr for attr in decision.selected_attributes if attr in available_attributes]
        return {
            "selected_attributes": _dedup_keep_order(selected),
            "resolution_mode": "semantic_llm_match",
            "available_attributes": available_attributes,
            "llm_decision": decision.model_dump() if hasattr(decision, "model_dump") else decision.dict(),
        }
    except Exception as exc:
        return {
            "selected_attributes": [],
            "resolution_mode": "semantic_llm_match_failed",
            "available_attributes": available_attributes,
            "error": str(exc),
        }


def make_clarify_prompt(context: str, missing: str) -> List[tuple]:
    if missing == "equipment":
        goal = "Kläre, welches Equipment gemeint ist und möglichst welchen Namen oder welche Nummer."
    elif missing == "state":
        goal = "Kläre, welche StateVariable gemeint ist."
    elif missing == "metric":
        goal = "Kläre, welche Metrik gemeint ist: P (Wirkleistung), Q (Blindleistung) oder S (Scheinleistung)."
    else:
        goal = "Kläre die Anfrage so, dass Equipment und State bestimmt werden können."

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
# 5) JSON Helpers
# =============================================================================

def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("No JSON object found in LLM output.")


# =============================================================================
# 6) Candidate shortlist aus network_index
# =============================================================================

def shortlist_candidates(
    user_input: str,
    network_index: dict,
    equipment_type: str,
    limit: int = 30,
    cutoff: float = 0.65,
) -> List[str]:
    equipment_name_index = (network_index or {}).get("equipment_name_index", {})
    name_index: Dict[str, Any] = equipment_name_index.get(equipment_type, {}) or {}
    keys = list(name_index.keys())
    if not keys:
        return []

    user_norm = normalize_text(user_input)
    scored: List[Tuple[int, str]] = []

    for k in keys:
        if k and k in user_norm:
            scored.append((1000 + len(k), k))

    nums = extract_two_numbers(user_input)
    if nums:
        n1, n2 = nums
        for k in keys:
            if n1 in k and n2 in k:
                scored.append((900 + len(k), k))

    n = extract_one_number(user_input)
    if n:
        for k in keys:
            if _number_boundary_match(n, k):
                scored.append((800 + len(k), k))

    fuzzy = get_close_matches(user_norm, keys, n=min(10, len(keys)), cutoff=cutoff)
    for k in fuzzy:
        scored.append((700 + len(k), k))

    if not scored:
        return sorted(keys, key=len, reverse=True)[: min(limit, len(keys))]

    seen: Set[str] = set()
    out: List[str] = []
    for _, k in sorted(scored, key=lambda x: x[0], reverse=True):
        if k not in seen:
            out.append(k)
            seen.add(k)
        if len(out) >= limit:
            break
    return out


def llm_choose_equipment_key(
    llm,
    *,
    user_context: str,
    equipment_type: str,
    candidate_keys: List[str],
) -> CandidateChoice:
    candidates_block = "\n".join(f"- {k}" for k in candidate_keys)

    user_prompt = f"""
Equipment-Typ:
{equipment_type}

Gesamter Dialogkontext:
{user_context}

candidate_keys:
{candidates_block}
""".strip()

    try:
        resp = llm.invoke([("system", SYSTEM_PROMPT_SELECT), ("user", user_prompt)])
        text = getattr(resp, "content", str(resp))
        data = extract_json(text)
        choice = CandidateChoice(**data)
    except Exception:
        return CandidateChoice(
            equipment_key=None,
            need_clarification=True,
            clarification_question=f"Welches konkrete {equipment_type}-Equipment meinst du (Name oder Nummer)?",
        )

    if choice.equipment_key is not None and choice.equipment_key not in candidate_keys:
        return CandidateChoice(
            equipment_key=None,
            need_clarification=True,
            clarification_question=f"Welches konkrete {equipment_type}-Equipment meinst du (Name oder Nummer)?",
        )

    return choice


# =============================================================================
# 7) User interaction
# =============================================================================

def default_ask_user(question: str) -> str:
    return input(f"{question}\n> ").strip()


# =============================================================================
# 8) Zeit-Helpers
# =============================================================================

def _parse_yyyy_mm_dd(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_dd_mm_yyyy(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    m = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", s)
    if not m:
        return None
    try:
        return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)), tzinfo=timezone.utc)
    except ValueError:
        return None


def _time_window_from_text(text: str) -> tuple[Optional[datetime], Optional[datetime], Optional[str]]:
    t = (text or "").lower()

    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", t)
    if len(iso_dates) >= 2:
        d1 = _parse_yyyy_mm_dd(iso_dates[0])
        d2 = _parse_yyyy_mm_dd(iso_dates[1])
        if d1 and d2:
            start = d1.replace(hour=0, minute=0, second=0, microsecond=0)
            end = d2.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return start, end, f"zwischen {start.date().isoformat()} und {d2.date().isoformat()}"

    de_dates = re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", t)
    if len(de_dates) >= 2:
        d1 = _parse_dd_mm_yyyy(de_dates[0])
        d2 = _parse_dd_mm_yyyy(de_dates[1])
        if d1 and d2:
            start = d1.replace(hour=0, minute=0, second=0, microsecond=0)
            end = d2.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return start, end, f"zwischen {start.date().isoformat()} und {d2.date().isoformat()}"

    d = _parse_yyyy_mm_dd(t)
    if d:
        start = d.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, f"am {start.date().isoformat()}"

    d = _parse_dd_mm_yyyy(t)
    if d:
        start = d.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, f"am {start.date().isoformat()}"

    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "vorgestern" in t:
        return today - timedelta(days=2), today - timedelta(days=1), "vorgestern"
    if "gestern" in t:
        return today - timedelta(days=1), today, "gestern"
    if "heute" in t:
        return today, today + timedelta(days=1), "heute"

    return None, None, None


def _ensure_time_window(
    context: str,
    ask_user: Callable[[str], str],
) -> tuple[Optional[datetime], Optional[datetime], Optional[str], str]:
    start, end, label = _time_window_from_text(context)
    final_context = context

    if start and end:
        return start, end, label, final_context

    q = "Bitte gib den genauen Zeitraum an (z.B. 'heute', 'gestern' oder ein Datum wie '2026-03-02')."
    a = (ask_user(q) or "").strip()
    if a:
        final_context = final_context + f"\nAssistant: {q}\nUser: {a}"
        start2, end2, label2 = _time_window_from_text(a)
        return start2, end2, (label2 or label), final_context

    return None, None, label, final_context




# =============================================================================
# 9a) interpret_equipment_type_query
# =============================================================================
def interpret_equipment_type_query(
    user_input: str,
    *,
    network_index: dict,
    ask_user: Optional[Callable[[str], str]] = None,
    max_rounds: int = 4,
    allowed_equipment_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Resolve only the CIM equipment *type* from a natural-language query.

    This is a dedicated path for type-only requests like
    'Welche PowerTransformer gibt es?' and intentionally does *not*
    resolve a concrete equipment instance.
    """
    if ask_user is None:
        ask_user = default_ask_user

    llm = get_llm()
    equipment_name_index = (network_index or {}).get("equipment_name_index", {}) or {}

    effective_allowed_equipment_types = _normalize_allowed_set(
        allowed_equipment_types or list(equipment_name_index.keys()) or (network_index or {}).get("equipment_types", []),
        set(equipment_name_index.keys()) or set((network_index or {}).get("equipment_types", [])) or DEFAULT_EQUIPMENT_TYPES,
    )

    system_prompt_parse = build_system_prompt_parse(
        allowed_equipment_types=effective_allowed_equipment_types,
        allowed_state_types=set(),
    )

    context_lines: List[str] = [f"User: {user_input}"]

    def parse_types_with_llm(context: str) -> Optional[QueryParse]:
        resp = llm.invoke([("system", system_prompt_parse), ("user", context)])
        text = getattr(resp, "content", str(resp))
        data = extract_json(text)
        parsed = QueryParse(**data)
        return normalize_query(
            parsed,
            allowed_equipment_types=effective_allowed_equipment_types,
            allowed_state_types=set(),
        )

    for _ in range(max_rounds):
        context = "\n".join(context_lines)

        try:
            parsed = parse_types_with_llm(context)
        except (ValidationError, ValueError, json.JSONDecodeError):
            parsed = None

        if parsed is None or not parsed.equipment_detected:
            q = llm.invoke(make_clarify_prompt(context, "equipment")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        selected_type = parsed.equipment_detected[0]
        return {
            "equipment_detected": list(parsed.equipment_detected),
            "selected_type": selected_type,
            "time_start": None,
            "time_end": None,
            "time_label": None,
            "mode": "equipment_type_only",
        }

    return {
        "equipment_detected": [],
        "selected_type": None,
        "time_start": None,
        "time_end": None,
        "time_label": None,
        "mode": "equipment_type_only",
    }


# =============================================================================
# 9) interpret_user_query
# =============================================================================

def interpret_user_query(
    user_input: str,
    *,
    network_index: dict,
    ask_user: Optional[Callable[[str], str]] = None,
    max_rounds: int = 8,
    default_state_if_equipment_only: str = "SvPowerFlow",
    require_time_window: bool = True,
    allowed_equipment_types: Optional[List[str]] = None,
    allowed_state_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if ask_user is None:
        ask_user = default_ask_user

    llm = get_llm()
    equipment_name_index = (network_index or {}).get("equipment_name_index", {}) or {}

    effective_allowed_equipment_types = _normalize_allowed_set(
        allowed_equipment_types or list(equipment_name_index.keys()),
        set(equipment_name_index.keys()) if equipment_name_index else DEFAULT_EQUIPMENT_TYPES,
    )

    effective_allowed_state_types = _normalize_allowed_set(
        allowed_state_types,
        DEFAULT_STATE_TYPES,
    )

    system_prompt_parse = build_system_prompt_parse(
        allowed_equipment_types=effective_allowed_equipment_types,
        allowed_state_types=effective_allowed_state_types,
    )

    context_lines: List[str] = [f"User: {user_input}"]

    def parse_types_with_llm(context: str) -> Optional[QueryParse]:
        resp = llm.invoke([("system", system_prompt_parse), ("user", context)])
        text = getattr(resp, "content", str(resp))
        data = extract_json(text)
        parsed = QueryParse(**data)
        return normalize_query(
            parsed,
            allowed_equipment_types=effective_allowed_equipment_types,
            allowed_state_types=effective_allowed_state_types,
        )

    for _ in range(max_rounds):
        context = "\n".join(context_lines)

        try:
            parsed = parse_types_with_llm(context)
        except (ValidationError, ValueError, json.JSONDecodeError):
            parsed = None

        if parsed is None:
            q = llm.invoke(make_clarify_prompt(context, "general")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        if parsed.equipment_detected and not parsed.state_detected and default_state_if_equipment_only in effective_allowed_state_types:
            parsed.state_detected = [default_state_if_equipment_only]
            parsed = normalize_query(
                parsed,
                allowed_equipment_types=effective_allowed_equipment_types,
                allowed_state_types=effective_allowed_state_types,
            )

        context_l = context.lower()
        explicit_metric = any(
            w in context_l for w in ["wirkleistung", "blindleistung", "scheinleistung", " p ", " q ", " s "]
        )

        if ("PowerTransformer" in parsed.equipment_detected) and (
            "auslastung" in context_l or "utilization" in context_l or "loading" in context_l
        ) and not explicit_metric:
            parsed.metric = "S"

        if parsed.metric is None and "SvPowerFlow" in parsed.state_detected:
            inferred_metric = infer_metric_from_context(llm, context)
            if inferred_metric in {"P", "Q", "S"}:
                parsed.metric = inferred_metric

        if parsed.equipment_detected and "SvPowerFlow" in parsed.state_detected:
            inferred_pf_equipment_type = infer_powerflow_equipment_type_from_context(
                llm,
                context,
                available_equipment_types=sorted(effective_allowed_equipment_types),
            )
            if inferred_pf_equipment_type in effective_allowed_equipment_types:
                parsed.equipment_detected = [inferred_pf_equipment_type]
                parsed = normalize_query(
                    parsed,
                    allowed_equipment_types=effective_allowed_equipment_types,
                    allowed_state_types=effective_allowed_state_types,
                )

        if parsed.equipment_detected and "SvVoltage" in parsed.state_detected:
            inferred_voltage_equipment_type = infer_voltage_equipment_type_from_context(
                llm,
                context,
                available_equipment_types=sorted(effective_allowed_equipment_types),
            )
            if inferred_voltage_equipment_type in effective_allowed_equipment_types:
                parsed.equipment_detected = [inferred_voltage_equipment_type]
                parsed = normalize_query(
                    parsed,
                    allowed_equipment_types=effective_allowed_equipment_types,
                    allowed_state_types=effective_allowed_state_types,
                )

        if not parsed.equipment_detected:
            q = llm.invoke(make_clarify_prompt(context, "equipment")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        if effective_allowed_state_types and not parsed.state_detected:
            q = llm.invoke(make_clarify_prompt(context, "state")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        selections: List[EquipmentSelection] = []
        clarification_needed = False

        for eq_type in parsed.equipment_detected:
            if eq_type not in effective_allowed_equipment_types:
                continue

            candidates = shortlist_candidates(
                user_input=context,
                network_index=network_index,
                equipment_type=eq_type,
                limit=30,
                cutoff=0.65,
            )

            if not candidates:
                q = f"Welches konkrete {eq_type}-Equipment meinst du (Name oder Nummer)?"
                a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
                context_lines += [f"Assistant: {q}", f"User: {a}"]
                clarification_needed = True
                break

            choice = llm_choose_equipment_key(
                llm=llm,
                user_context=context,
                equipment_type=eq_type,
                candidate_keys=candidates,
            )

            if choice.equipment_key is None:
                q = choice.clarification_question or f"Welches konkrete {eq_type}-Equipment meinst du (Name oder Nummer)?"
                a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
                context_lines += [f"Assistant: {q}", f"User: {a}"]
                clarification_needed = True
                break

            eq_obj = equipment_name_index.get(eq_type, {}).get(choice.equipment_key)
            selections.append(
                EquipmentSelection(
                    equipment_type=eq_type,
                    equipment_key=choice.equipment_key,
                    equipment_name=getattr(eq_obj, "name", None) if eq_obj is not None else None,
                    equipment_id=equipment_identifier(eq_obj) if eq_obj is not None else None,
                )
            )

        if clarification_needed:
            continue

        if require_time_window:
            start_dt, end_dt, label, _ = _ensure_time_window(context, ask_user)
            if start_dt and end_dt:
                parsed.time_start = start_dt.isoformat()
                parsed.time_end = end_dt.isoformat()
                parsed.time_label = label

        parsed.equipment_selection = selections
        parsed = normalize_query(
            parsed,
            allowed_equipment_types=effective_allowed_equipment_types,
            allowed_state_types=effective_allowed_state_types,
        )

        return parsed.model_dump()

    return {
        "equipment_detected": [],
        "state_detected": [],
        "metric": None,
        "equipment_selection": [],
        "time_start": None,
        "time_end": None,
        "time_label": None,
    }

