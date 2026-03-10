from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError

from cimpy.cimpy_time_analysis.langchain_llm import get_llm


# =============================================================================
# Planner Schema
# =============================================================================

QueryMode = Literal[
    "topology",
    "state",
    "topology_plus_state",
    "unknown",
]

AggregationType = Optional[Literal[
    "max",
    "min",
    "mean",
    "sum",
    "none",
]]

TopologyScope = Optional[Literal[
    "neighbors",
    "component",
    "path",
    "none",
]]


class AnalysisPlan(BaseModel):
    """
    Grober Analyseplan, der VOR dem eigentlichen Laden erstellt wird.

    Wichtig:
    - bewusst tool-/MCP-freundlich
    - keine echten Python-Objekte, nur primitive/JSON-nahe Felder
    """

    query_mode: QueryMode = "unknown"

    # Welche Datenpakete werden gebraucht?
    needs_structure: bool = True
    needs_topology_graph: bool = False
    needs_state_types: List[str] = Field(default_factory=list)

    # Zielobjekte / fachliche Einschränkung
    target_equipment_types: List[str] = Field(default_factory=list)

    # Zeitlogik
    requires_time_window: bool = True
    time_hint: Optional[str] = None

    # Semantik der Frage
    topology_scope: TopologyScope = "none"
    aggregation: AggregationType = "none"
    metric_hint: Optional[str] = None

    # Zusatzinfos für spätere Tool-Steuerung
    graph_level: Literal["connectivity", "topological"] = "connectivity"

    # Debug / Steuerung
    confidence: float = 0.0
    reasoning_short: Optional[str] = None


# =============================================================================
# JSON Helpers
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


def _safe_list_of_strings(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]

    out = []
    seen = set()
    for x in value:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# =============================================================================
# Inventory / Context Compression
# =============================================================================

def summarize_inventory_for_planner(snapshot_inventory: Optional[dict]) -> Dict[str, Any]:
    """
    Kompakte, LLM-taugliche Zusammenfassung des Snapshot-Inventars.
    Wir geben absichtlich nur wenige, aber nützliche Infos weiter.
    """
    snapshots = (snapshot_inventory or {}).get("snapshots", []) or []

    if not snapshots:
        return {
            "num_snapshots": 0,
            "first_snapshot_name": None,
            "last_snapshot_name": None,
            "time_min": None,
            "time_max": None,
            "has_sv_profile": False,
            "has_tp_profile": False,
            "has_eq_profile": False,
            "has_ssh_profile": False,
        }

    time_values = [s.get("default_time_str") for s in snapshots if s.get("default_time_str")]

    return {
        "num_snapshots": len(snapshots),
        "first_snapshot_name": snapshots[0].get("snapshot_name"),
        "last_snapshot_name": snapshots[-1].get("snapshot_name"),
        "time_min": min(time_values) if time_values else None,
        "time_max": max(time_values) if time_values else None,
        "has_sv_profile": any(bool(s.get("has_sv_profile")) for s in snapshots),
        "has_tp_profile": any(bool(s.get("has_tp_profile")) for s in snapshots),
        "has_eq_profile": any(bool(s.get("has_eq_profile")) for s in snapshots),
        "has_ssh_profile": any(bool(s.get("has_ssh_profile")) for s in snapshots),
    }


def get_available_equipment_types_from_index(network_index: Optional[dict]) -> List[str]:
    equipment_name_index = (network_index or {}).get("equipment_name_index", {}) or {}
    return sorted(str(k) for k in equipment_name_index.keys())


# =============================================================================
# Heuristics (Fallback + leichte Stabilisierung)
# =============================================================================

def _detect_aggregation_heuristic(user_input: str) -> str:
    text = (user_input or "").lower()

    if any(w in text for w in ["höchste", "hoechste", "maximum", "maximal", "größte", "groesste", "max "]):
        return "max"
    if any(w in text for w in ["niedrigste", "kleinste", "minimum", "minimal", "min "]):
        return "min"
    if any(w in text for w in ["mittel", "durchschnitt", "average", "mean"]):
        return "mean"
    if any(w in text for w in ["summe", "gesamt", "aufsummiert"]):
        return "sum"

    return "none"


def _detect_topology_scope_heuristic(user_input: str) -> str:
    text = (user_input or "").lower()

    if any(w in text for w in ["nachbar", "nachbarn", "direkt verbunden", "angeschlossen", "hängt an"]):
        return "neighbors"
    if any(w in text for w in ["komponente", "zusammenhäng", "zusammenhaeng", "insel", "teilnetz", "netzsegment"]):
        return "component"
    if any(w in text for w in ["pfad", "weg", "route", "verbindung zwischen"]):
        return "path"

    return "none"


def _detect_graph_level_heuristic(user_input: str) -> str:
    text = (user_input or "").lower()

    if any(w in text for w in ["topologicalnode", "topological node", "topologisch", "elektrisch zusammenhängend", "elektrisch zusammenhaengend"]):
        return "topological"

    return "connectivity"


def _detect_state_types_heuristic(user_input: str) -> List[str]:
    text = (user_input or "").lower()
    out = []

    if any(w in text for w in ["spannung", "voltage", "kv", "svvoltage"]):
        out.append("SvVoltage")

    if any(w in text for w in [
        "leistung", "wirkleistung", "blindleistung", "scheinleistung",
        "auslastung", "loading", "utilization", "mw", "mvar", "mva", "svpowerflow"
    ]):
        out.append("SvPowerFlow")

    return out


def _detect_equipment_types_heuristic(user_input: str, available_equipment_types: List[str]) -> List[str]:
    text = (user_input or "").lower()
    detected = []

    mapping = {
        "PowerTransformer": ["trafo", "transformator", "powertransformer"],
        "ConformLoad": ["load", "last", "verbraucher", "conformload"],
        "ACLineSegment": ["leitung", "line", "aclinesegment"],
        "Breaker": ["schalter", "breaker"],
        "Disconnector": ["trenner", "disconnector"],
        "BusbarSection": ["busbar", "sammelschiene", "busbarsection"],
        "SynchronousMachine": ["generator", "maschine", "synchronousmachine"],
    }

    for eq_type in available_equipment_types:
        hints = mapping.get(eq_type, [eq_type.lower()])
        if any(h in text for h in hints):
            detected.append(eq_type)

    return detected


def _infer_query_mode(
    *,
    needs_topology_graph: bool,
    needs_state_types: List[str],
) -> str:
    if needs_topology_graph and needs_state_types:
        return "topology_plus_state"
    if needs_topology_graph:
        return "topology"
    if needs_state_types:
        return "state"
    return "unknown"


def _normalize_analysis_plan(data: Dict[str, Any], user_input: str, available_equipment_types: List[str]) -> Dict[str, Any]:
    """
    Härtet LLM-Output mit deterministischen Defaults / Heuristiken.
    """
    data = dict(data or {})

    data["needs_state_types"] = _safe_list_of_strings(data.get("needs_state_types"))
    data["target_equipment_types"] = _safe_list_of_strings(data.get("target_equipment_types"))

    # nur tatsächlich verfügbare Equipment-Typen zulassen
    if available_equipment_types:
        data["target_equipment_types"] = [
            t for t in data["target_equipment_types"]
            if t in available_equipment_types
        ]

    # Fallbacks aus Heuristik
    if not data["target_equipment_types"]:
        data["target_equipment_types"] = _detect_equipment_types_heuristic(
            user_input=user_input,
            available_equipment_types=available_equipment_types,
        )

    if not data["needs_state_types"]:
        data["needs_state_types"] = _detect_state_types_heuristic(user_input)

    if not data.get("aggregation"):
        data["aggregation"] = _detect_aggregation_heuristic(user_input)

    if not data.get("topology_scope"):
        data["topology_scope"] = _detect_topology_scope_heuristic(user_input)

    if not data.get("graph_level"):
        data["graph_level"] = _detect_graph_level_heuristic(user_input)

    # Bool-Fallbacks
    if "needs_structure" not in data:
        data["needs_structure"] = True

    if "needs_topology_graph" not in data:
        data["needs_topology_graph"] = data.get("topology_scope") not in [None, "none"]

    if "requires_time_window" not in data:
        # Für historische State-Fragen meist ja, für reine Topologie nein
        data["requires_time_window"] = bool(data["needs_state_types"])

    # Query Mode ableiten
    if not data.get("query_mode") or data.get("query_mode") == "unknown":
        data["query_mode"] = _infer_query_mode(
            needs_topology_graph=bool(data.get("needs_topology_graph")),
            needs_state_types=data.get("needs_state_types", []),
        )

    # metric_hint normalisieren
    metric_hint = data.get("metric_hint")
    if metric_hint is not None:
        metric_hint = str(metric_hint).strip().upper() or None
        if metric_hint not in {"P", "Q", "S"}:
            metric_hint = None
    data["metric_hint"] = metric_hint

    # aggregation normalisieren
    aggregation = str(data.get("aggregation") or "none").strip().lower()
    if aggregation not in {"max", "min", "mean", "sum", "none"}:
        aggregation = "none"
    data["aggregation"] = aggregation

    # topology_scope normalisieren
    topology_scope = str(data.get("topology_scope") or "none").strip().lower()
    if topology_scope not in {"neighbors", "component", "path", "none"}:
        topology_scope = "none"
    data["topology_scope"] = topology_scope

    # graph_level normalisieren
    graph_level = str(data.get("graph_level") or "connectivity").strip().lower()
    if graph_level not in {"connectivity", "topological"}:
        graph_level = "connectivity"
    data["graph_level"] = graph_level

    # confidence
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    data["confidence"] = confidence

    # reasoning_short
    rs = data.get("reasoning_short")
    data["reasoning_short"] = str(rs).strip() if rs is not None else None

    return data


# =============================================================================
# Prompting
# =============================================================================

PLANNER_SYSTEM_PROMPT = """
Du planst Datenzugriffe für historische CIM-Analysen.

Deine Aufgabe:
- Bestimme NICHT das konkrete Equipment.
- Erstelle nur einen groben Analyseplan, damit später gezielt geladen werden kann.
- Antworte AUSSCHLIESSLICH mit JSON.
- Kein Markdown, kein Zusatztext.

Wichtige Regeln:
1) Wenn die Frage topologisch ist, setze needs_topology_graph=true.
2) Wenn die Frage Spannungen betrifft, nutze needs_state_types=["SvVoltage"].
3) Wenn die Frage Leistungen / Auslastung betrifft, nutze needs_state_types=["SvPowerFlow"].
4) Bei Kombinationen aus Topologie + Zustand nutze query_mode="topology_plus_state".
5) target_equipment_types nur aus den verfügbaren Typen wählen.
6) Wenn kein Zeitraum explizit genannt ist, darf requires_time_window trotzdem true sein, falls State-Daten gebraucht werden.
7) metric_hint nur P, Q, S oder null.
8) topology_scope nur "neighbors", "component", "path" oder "none".
9) aggregation nur "max", "min", "mean", "sum" oder "none".
10) graph_level nur "connectivity" oder "topological".

Schema:
{
  "query_mode": "topology" | "state" | "topology_plus_state" | "unknown",
  "needs_structure": true | false,
  "needs_topology_graph": true | false,
  "needs_state_types": ["SvVoltage" | "SvPowerFlow", ...],
  "target_equipment_types": ["<equipment type>", ...],
  "requires_time_window": true | false,
  "time_hint": "<kurzer Hinweis aus der Frage oder null>",
  "topology_scope": "neighbors" | "component" | "path" | "none",
  "aggregation": "max" | "min" | "mean" | "sum" | "none",
  "metric_hint": "P" | "Q" | "S" | null,
  "graph_level": "connectivity" | "topological",
  "confidence": 0.0,
  "reasoning_short": "<sehr kurz>"
}
""".strip()


def build_planner_prompt(
    *,
    user_input: str,
    inventory_summary: Dict[str, Any],
    available_equipment_types: List[str],
) -> str:
    return f"""
Nutzerfrage:
{user_input}

Verfügbare Equipment-Typen:
{available_equipment_types}

Snapshot-Inventar (kompakt):
{inventory_summary}

Erstelle den Analyseplan nur auf dieser Basis.
""".strip()


# =============================================================================
# Public API
# =============================================================================

def plan_analysis(
    user_input: str,
    *,
    snapshot_inventory: Optional[dict] = None,
    network_index: Optional[dict] = None,
    llm=None,
) -> Dict[str, Any]:
    """
    Hauptfunktion für den vorgeschalteten Analyseplan.

    Input:
    - user_input
    - snapshot_inventory (optional, aber empfohlen)
    - network_index (optional, um verfügbare Equipment-Typen zu kennen)

    Output:
    - dict im AnalysisPlan-Schema
    """
    if llm is None:
        llm = get_llm()

    inventory_summary = summarize_inventory_for_planner(snapshot_inventory)
    available_equipment_types = get_available_equipment_types_from_index(network_index)

    prompt = build_planner_prompt(
        user_input=user_input,
        inventory_summary=inventory_summary,
        available_equipment_types=available_equipment_types,
    )

    raw = {}
    llm_error = None

    try:
        response = llm.invoke([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("user", prompt),
        ])
        content = getattr(response, "content", str(response))
        raw = extract_json(content)
    except Exception as e:
        llm_error = str(e)
        raw = {}

    normalized = _normalize_analysis_plan(
        data=raw,
        user_input=user_input,
        available_equipment_types=available_equipment_types,
    )

    # Falls das LLM komplett ausfällt, trotzdem gültiges Schema liefern
    if llm_error and not normalized.get("reasoning_short"):
        normalized["reasoning_short"] = f"fallback_after_llm_error: {llm_error}"

    try:
        plan = AnalysisPlan(**normalized)
    except ValidationError:
        # Letzte harte Fallback-Stufe
        fallback = AnalysisPlan(
            query_mode=_infer_query_mode(
                needs_topology_graph=bool(normalized.get("needs_topology_graph")),
                needs_state_types=_safe_list_of_strings(normalized.get("needs_state_types")),
            ),
            needs_structure=bool(normalized.get("needs_structure", True)),
            needs_topology_graph=bool(normalized.get("needs_topology_graph", False)),
            needs_state_types=_safe_list_of_strings(normalized.get("needs_state_types")),
            target_equipment_types=_safe_list_of_strings(normalized.get("target_equipment_types")),
            requires_time_window=bool(normalized.get("requires_time_window", False)),
            time_hint=normalized.get("time_hint"),
            topology_scope=normalized.get("topology_scope") if normalized.get("topology_scope") in {"neighbors", "component", "path", "none"} else "none",
            aggregation=normalized.get("aggregation") if normalized.get("aggregation") in {"max", "min", "mean", "sum", "none"} else "none",
            metric_hint=normalized.get("metric_hint") if normalized.get("metric_hint") in {"P", "Q", "S", None} else None,
            graph_level=normalized.get("graph_level") if normalized.get("graph_level") in {"connectivity", "topological"} else "connectivity",
            confidence=float(normalized.get("confidence", 0.0) or 0.0),
            reasoning_short=normalized.get("reasoning_short"),
        )
        return fallback.model_dump()

    return plan.model_dump()


def plan_analysis_with_debug(
    user_input: str,
    *,
    snapshot_inventory: Optional[dict] = None,
    network_index: Optional[dict] = None,
    llm=None,
) -> Dict[str, Any]:
    """
    Komfortfunktion mit zusätzlichem Debug-Kontext.
    Praktisch für spätere Tool-/MCP-Nutzung.
    """
    inventory_summary = summarize_inventory_for_planner(snapshot_inventory)
    available_equipment_types = get_available_equipment_types_from_index(network_index)

    plan = plan_analysis(
        user_input=user_input,
        snapshot_inventory=snapshot_inventory,
        network_index=network_index,
        llm=llm,
    )

    return {
        "plan": plan,
        "context": {
            "inventory_summary": inventory_summary,
            "available_equipment_types": available_equipment_types,
        },
    }