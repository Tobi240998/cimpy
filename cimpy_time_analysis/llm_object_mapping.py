from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import List, Optional, Literal, Set, Dict, Any, Callable, Tuple

from pydantic import BaseModel, Field, ValidationError
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
    """
    Normalisierung für Matching:
    - lower
    - transformator -> trafo
    - trf -> trafo
    - / -> -
    - Leerzeichen entfernen
    """
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
    """
    Versucht eine stabile ID zu finden.
    """
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

Metric = Optional[Literal["P", "Q", "S"]]


class EquipmentSelection(BaseModel):
    equipment_type: str
    # key ist der normalisierte Name-Key aus network_index["equipment_name_index"][type].keys()
    equipment_key: str
    equipment_name: Optional[str] = None
    equipment_id: Optional[str] = None


class QueryParse(BaseModel):
    equipment_detected: List[str] = Field(default_factory=list)
    state_detected: List[str] = Field(default_factory=list)
    metric: Metric = None

    equipment_selection: List[EquipmentSelection] = Field(default_factory=list)

    # Zeitfenster (ISO-Strings in UTC), optional
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    time_label: Optional[str] = None


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
- time_start und time_end müssen, wenn möglich, als UTC-ISO-Strings ausgegeben werden.
- Für "am 2026-01-09" gilt:
  - time_start = "2026-01-09T00:00:00+00:00"
  - time_end   = "2026-01-10T00:00:00+00:00"
- Für "09.01.2026" gilt dasselbe Datum wie für "2026-01-09".
- Für "zwischen 09.01.2026 und 11.01.2026" gilt:
  - start = Beginn des ersten Tages
  - end   = Beginn des Tages NACH dem letzten Datum
- Für "vom 09.01.2026 bis 11.01.2026" gilt dasselbe.
- Für "gestern" / "heute" darfst du relative Zeiträume interpretieren.
- time_label soll z.B. sein:
  - "am 2026-01-09"
  - "zwischen 2026-01-09 und 2026-01-11"
  - "gestern"

Synonyme:
- Trafo/Transformator/Transformer => PowerTransformer
- Verbraucher/Last/Load/ConformLoad => ConformLoad
- Spannung/Voltage => SvVoltage
- Leistung/Power/Powerflow => SvPowerFlow
- Wirkleistung/P => metric "P"
- Blindleistung/Q => metric "Q"
- Scheinleistung/S => metric "S"
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
- equipment_key muss exakt einem candidate_key entsprechen, wenn du sicher bist.
- Wenn nicht sicher: equipment_key=null, need_clarification=true und stelle eine kurze Rückfrage.
""".strip()


CLARIFY_SYSTEM = "Du stellst genau EINE kurze Rückfrage auf Deutsch. Kein Zusatztext."


def make_clarify_prompt(context: str, missing: str) -> List[tuple]:
    if missing == "equipment":
        goal = "Kläre, welches Equipment gemeint ist und möglichst welchen Namen oder welche Nummer."
    elif missing == "state":
        goal = "Kläre, welche StateVariable gemeint ist."
    elif missing == "metric":
        goal = "Kläre welche Metrik gemeint ist: P (Wirkleistung), Q (Blindleistung) oder S (Scheinleistung)."
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
    """
    Liefert candidate_keys = normalisierte Namen-Keys aus equipment_name_index[equipment_type].
    Priorisierung:
      1) direct substring in user_norm
      2) two-number match
      3) one-number boundary match
      4) fuzzy match
    """
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
        fallback = sorted(keys, key=len, reverse=True)[: min(limit, len(keys))]
        return fallback

    seen: Set[str] = set()
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    out: List[str] = []
    for _, k in scored_sorted:
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
) -> Dict[str, Any]:
    """
    LLM soll nur aus candidate_keys auswählen oder Rückfrage stellen.
    """
    candidates_block = "\n".join(f"- {k}" for k in candidate_keys)

    user_prompt = f"""
Equipment-Typ: {equipment_type}

Kontext:
{user_context}

candidate_keys (du MUSST exakt einen davon wählen, wenn sicher):
{candidates_block}
""".strip()

    resp = llm.invoke([("system", SYSTEM_PROMPT_SELECT), ("user", user_prompt)])
    text = getattr(resp, "content", str(resp))
    data = extract_json(text)

    ek = data.get("equipment_key", None)
    if ek is not None and ek not in candidate_keys:
        return {
            "equipment_key": None,
            "need_clarification": True,
            "clarification_question": "Welches konkrete Equipment meinst du (Name/Nummer)?",
        }

    return {
        "equipment_key": ek,
        "need_clarification": bool(data.get("need_clarification", False)) if ek is None else False,
        "clarification_question": data.get("clarification_question", None),
    }


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
    """
    Minimal robust:
    - "heute" / "gestern" / "vorgestern"
    - explizites Datum YYYY-MM-DD
    - explizites Datum DD.MM.YYYY
    - Bereiche "zwischen ... und ..."
    - Bereiche "vom ... bis ..."
    Zeitfenster ist [start, end) in UTC.
    """
    t = (text or "").lower()

    # Bereich mit ISO-Daten
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", t)
    if len(iso_dates) >= 2:
        d1 = _parse_yyyy_mm_dd(iso_dates[0])
        d2 = _parse_yyyy_mm_dd(iso_dates[1])
        if d1 and d2:
            start = d1.replace(hour=0, minute=0, second=0, microsecond=0)
            end = d2.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return start, end, f"zwischen {start.date().isoformat()} und {d2.date().isoformat()}"

    # Bereich mit deutschem Datumsformat
    de_dates = re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", t)
    if len(de_dates) >= 2:
        d1 = _parse_dd_mm_yyyy(de_dates[0])
        d2 = _parse_dd_mm_yyyy(de_dates[1])
        if d1 and d2:
            start = d1.replace(hour=0, minute=0, second=0, microsecond=0)
            end = d2.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return start, end, f"zwischen {start.date().isoformat()} und {d2.date().isoformat()}"

    # einzelnes ISO-Datum
    d = _parse_yyyy_mm_dd(t)
    if d:
        start = d.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, f"am {start.date().isoformat()}"

    # einzelnes deutsches Datum
    d = _parse_dd_mm_yyyy(t)
    if d:
        start = d.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, f"am {start.date().isoformat()}"

    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "vorgestern" in t:
        start = today - timedelta(days=2)
        end = today - timedelta(days=1)
        return start, end, "vorgestern"

    if "gestern" in t:
        start = today - timedelta(days=1)
        end = today
        return start, end, "gestern"

    if "heute" in t:
        start = today
        end = today + timedelta(days=1)
        return start, end, "heute"

    if "über den tag" in t or "ueber den tag" in t or "ganzen tag" in t:
        return None, None, "ganzer_tag"

    return None, None, None


def _ensure_time_window(
    context: str,
    ask_user: Callable[[str], str],
) -> tuple[Optional[datetime], Optional[datetime], Optional[str], str]:
    """
    Gibt (start_dt, end_dt, label, final_context) zurück.
    Wenn nicht extrahierbar, fragt nach (einmal) und hängt die Antwort an den Kontext an.
    """
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
    """
    Liefert IMMER:
    {
      "equipment_detected": [...],
      "state_detected": [...],
      "metric": "P|Q|S"|None,
      "equipment_selection": [
         {"equipment_type": "...", "equipment_key": "...", "equipment_name": "...", "equipment_id": "..."},
         ...
      ],
      "time_start": "...",
      "time_end": "...",
      "time_label": "..."
    }

    Neu:
    - allowed_equipment_types: optionale Einschränkung durch Planner
    - allowed_state_types: optionale Einschränkung durch Planner
    - require_time_window bleibt steuerbar

    Policy:
    - Equipment + State sollen am Ende vorhanden sein, sofern die Anfrage State-Daten braucht.
    - Wenn Equipment erkannt aber State fehlt => default SvPowerFlow (oder Planner-Vorgabe).
    - Wenn Equipment erkannt => wir versuchen zu jeder equipment_type ein konkretes Objekt zu wählen.
    - Wenn nicht eindeutig => Rückfrage-Schleife.
    """
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

        parsed: Optional[QueryParse] = None
        try:
            parsed = parse_types_with_llm(context)
        except (ValidationError, ValueError, json.JSONDecodeError):
            parsed = None

        if parsed is None:
            q = llm.invoke(make_clarify_prompt(context, "general")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        # Default-State nur setzen, wenn State-Typen überhaupt erlaubt sind
        if parsed.equipment_detected and not parsed.state_detected and default_state_if_equipment_only in effective_allowed_state_types:
            parsed.state_detected = [default_state_if_equipment_only]
            parsed = normalize_query(
                parsed,
                allowed_equipment_types=effective_allowed_equipment_types,
                allowed_state_types=effective_allowed_state_types,
            )

        # Heuristik: "Auslastung" eines Trafos => standardmäßig S
        context_l = context.lower()
        explicit_metric = any(
            w in context_l for w in ["wirkleistung", "blindleistung", "scheinleistung", " p ", " q ", " s "]
        )

        if ("PowerTransformer" in parsed.equipment_detected) and (
            "auslastung" in context_l or "utilization" in context_l or "loading" in context_l
        ):
            if not explicit_metric:
                parsed.metric = "S"

        # Equipment muss da sein
        if not parsed.equipment_detected:
            q = llm.invoke(make_clarify_prompt(context, "equipment")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        # State nur erzwingen, wenn State-Typen erlaubt/gewünscht sind
        if effective_allowed_state_types and not parsed.state_detected:
            q = llm.invoke(make_clarify_prompt(context, "state")).content.strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        # Konkretes Equipment wählen
        selections: List[EquipmentSelection] = []
        need_more_clarification = False
        clarification_question = None

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
                need_more_clarification = True
                clarification_question = f"Ich finde keine {eq_type}-Namen im Index. Welches konkrete Equipment meinst du (Name/Nummer)?"
                break

            choice = llm_choose_equipment_key(
                llm,
                user_context=context,
                equipment_type=eq_type,
                candidate_keys=candidates,
            )

            if choice["equipment_key"] is None:
                need_more_clarification = True
                clarification_question = choice.get("clarification_question") or "Welches konkrete Equipment meinst du (Name/Nummer)?"
                break

            equipment_key = choice["equipment_key"]
            eq_obj = equipment_name_index.get(eq_type, {}).get(equipment_key)

            sel = EquipmentSelection(
                equipment_type=eq_type,
                equipment_key=equipment_key,
                equipment_name=getattr(eq_obj, "name", None) if eq_obj is not None else None,
                equipment_id=equipment_identifier(eq_obj) if eq_obj is not None else None,
            )
            selections.append(sel)

        if need_more_clarification:
            q = (clarification_question or "Welches konkrete Equipment meinst du (Name/Nummer)?").strip()
            a = (ask_user(q) or "").strip() or "Ich bin mir nicht sicher."
            context_lines += [f"Assistant: {q}", f"User: {a}"]
            continue

        # Zeitraum optional erzwingen
        if require_time_window:
            start_dt, end_dt, label, context2 = _ensure_time_window(context, ask_user)
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


# =============================================================================
# 10) Standalone-Debug-Placeholder
# =============================================================================

class LLM_resultAgent:
    pass


def handle_user_query(user_input, snapshot_cache, network_index):
    parsed = interpret_user_query(user_input, network_index=network_index)

    equipment_detected = parsed.get("equipment_detected", [])
    state_detected = parsed.get("state_detected", [])
    metric = parsed.get("metric", None)
    equipment_obj = parsed.get("equipment_obj", [])

    agent = LLM_resultAgent()

    print("equipment_detected:", equipment_detected)
    print("state_detected:", state_detected)
    print("metric:", metric)
    print("equipment_obj:", equipment_obj)