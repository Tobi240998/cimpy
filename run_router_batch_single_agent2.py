from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.callbacks import get_usage_metadata_callback

from cimpy.single_agent2.llm_routing.orchestrator import Orchestrator


USER_INPUTS: List[str] = [
   "Welche Lasten gibt es in PowerFactory?",
    "Zeig mir alle verfügbaren Lasten im aktiven PowerFactory-Projekt.",
    "Liste die vorhandenen Loads in Powerfactory auf.",
    "Welche Verbraucher sind im PF-Modell vorhanden?",
    "Erhöhe Last A um 2 MW.",
    "Füge 2 MW bei Load A hinzu.",
    "Erhöhe Verbraucher A um 2 MW.",
    "Reduziere Last B um 3 MW.",
    "Ziehe 3 MW von Last B ab.",
    "Mache Load B um 3 MW kleiner.",
    "Erhöhe Last B um 3 MW. Wie ist die Auslastung von Line 4-5?",
    "Erhöhe Last B um 3 MW und gib die Auslastung der Leitung 4-5 aus.",
    "Last B 3 MW rauf, danach bitte die Belastung von Line 4-5.",
    "Reduziere Last A um 2 MW und gib danach die Auslastungen aller Leitungen zurück.",
    "Senke Last A um 2 MW. Wie hoch sind anschließend die Leitungsauslastungen?",
    "Last A minus 2 MW; bitte danach alle Leitungsbelastungen ausgeben.",
    "Öffne den Schalter Schalter 1.",
    "Schalte Schalter 1 aus.",
    "Trenne Switch 1.",
    "Schließe den Schalter Schalter 1.",
    "Schalte Schalter 1 ein.",
    "Zuschalten: Switch 1.",
    "Schalte den Schalter Schalter 1 um.",
    "Toggle Schalter 1.",
    "Welche direkten Nachbarn hat Last A in Powerfactory?",
    "Wer ist in PF direkt mit Load A verbunden?",
    "Zeig mir die unmittelbaren Nachbarn von Verbraucher A in Powerfactory.",
    "An welche Assets grenzt Last A in Powerfactory direkt?",
    "Welche direkten Nachbarn hat Bus 5 in PF?",
    "Was ist bei Powerfactory direkt mit Bus 5 verbunden?",
    "Zeig mir die unmittelbaren Nachbarn von Bus 5 in Powerfactory.",
    "Welche Assets hängen direkt an Bus 5 in Powerfactory?",
    "Wie ist die Nennspannung von Bus 1 in PowerFactory?",
    "Gib mir die Basisdaten-Nennspannung von Bus 1 aus Powerfactory.",
    "Was ist die rated voltage von Bus 1 bei Powerfactory?",
    "Was ist die obere Spannungsgrenze von Bus 1 in PowerFactory?",
    "Wie hoch ist die untere Spannungsgrenze von Bus 1 in Powerfactory?",
    "Nenne mir die Spannungsgrenzen von Bus 1 in Powerfactory.",
    "Welche Umin- und Umax-Grenzen sind bei Bus 1 in Powerfactory hinterlegt?",
    "Wie ist die Spannung von Bus 1 nach dem Lastfluss in PF?",
    "Gib mir die aktuelle Busspannung von Bus 1 als Lastflussergebnis aus PF.",
    "Was ist die Leiter-Leiter-Spannung von Bus 1 nach dem Loadflow in Powerfactory?",
    "Wie sind die Spannungen der Busse in PowerFactory?",
    "Gib mir die Spannungen aller Busse in PF.",
    "Welche Busspannungen gibt es nach dem Lastfluss in PF?",
    "Wie ist die Auslastung von Line 4-5 in PowerFactory?",
    "Wie hoch ist die Belastung von Line 4-5 nach dem Lastfluss in PF?",
    "Was ist die loading von Line 4-5 in PF?",
    "Wie sind die Auslastungen der Leitungen in PowerFactory?",
    "Welche Leitungsauslastungen liegen nach dem Lastfluss in PF vor?",
    "Zeige die Belastung für alle Lines in PF.",
    "Welche Attribute sind für Bus 1 in PowerFactory verfügbar?",
    "Welche Datenfelder gibt es bei Bus 1 in Powerfactory?",
    "Liste alle verfügbaren Attribute von Bus 1 in Powerfactory auf.",
    "Welche Attribute sind für Line 4-5 in Powerfactory verfügbar?",
    "Liste die verfügbaren Attribute von Leitung 4-5 in Powerfactory auf.",
    "Welche Datenfelder gibt es bei der Leitung 4-5 in Powerfactory?",
    "Lies in PF bei Bus 1 das Attribut vmin aus.",
    "Gib für Bus 1 in Powerfactory den Wert von vmax zurück.",
    "Lies in PF bei Line 4-5 das Attribut loading aus.",
    "Zeige für Bus 1 den Wert von uknom in Powerfactory.",
    "Was ist der Bemessungsfaktor von Trafo T1 in Powerfactory?",
    "Welche Länge hat Line 4-5 in Powerfactory?",
    "Gib mir den Widerstand von Line 4-5 in PF.",
    "Wie lang ist Leitung 4-5 in PF?",
    "Was ist die Reaktanz von Line 4-5 in PF?",
    "Gib mir die Nennspannungen aller Busse in Powerfactory.",
    "Welche rated voltages haben die Busse in Powerfactory?",
    "Zeige die Basisdaten-Nennspannung für alle Busse in Powerfactory.",
    "Nenne mir die Nennspannung sämtlicher Busse in Powerfactory.",
    "Welche Spannungsgrenzen haben die Busse in PF?",
    "Gib Umin und Umax für alle Busse in PF zurück.",
    "Zeige die oberen und unteren Spannungsgrenzen aller Busse in PF.",
    "Welche Voltage Limits sind für die Busse in Powerfactory hinterlegt?",
    "Öffne den Schalter Schalter 1 und gib danach die Busspannungen zurück.",
    "Schalte Schalter 1 aus. Wie sind anschließend die Busspannungen?",
    "Schließe den Schalter Schalter 1 und nenne danach die Netzspannungen.",
    "Wie is die spannung von bus 1 nachm lastfluss im Simulationsprogramm?",
    "Gib mir mal bitte die busspannungen nachm LF in PF.",
    "Mach load a 5 mw hoeher und sag busspannungen.",
    "Wie ist die Auslastung von Line in PF?",
    "Welche Farbe hat Bus 1 in Powerfactory?",
    "Wie ist die Spannung in PF?",
    "Schalte Schalter B an.",
    "Erhöhe Last 3 um 20 MW.",
    "Welche direkten topologischen Nachbarn hat Trafo 19-20 in den CIM-Daten?",
    "Was hängt direkt an Transformator 19/20 in den CIM-Daten?",
    "Welche Betriebsmittel sind in den CIM-Daten unmittelbar mit Trf 19 20 verbunden?",
    "Zu welcher zusammenhängenden Komponente gehört Trafo 19-20 bei den CIM-Daten?",
    "Zeig mir die Komponenten in den CIM-Daten rund um Transformator 19/20.",
    "Welche Betriebsmittel liegen in der topologischen Komponente von Trf 19 20 in den CIM-Daten?",
    "Wie hat sich am 09.01.2026 in den CIM-Daten die Spannung an Trafo 19-20 über die Zeit entwickelt?",
    "Zeig mir den Spannungsverlauf am 09.01.2026 von Transformator 19/20 aus den historischen Daten.",
    "Welche Spannung liegt im Zeitverlauf am 09.01.2026 an Trf 19 20 an?",
    "Gib mir Min, Max und Mittelwert der Spannung von Trafo 19-20 aus den CIM-Daten am 09.01.2026.",
    "Wie sind minimale, maximale und mittlere Spannung an Transformator 19/20 in den historischen Daten vom 09.01.2026?",
    "Nenne für Trf 19 20 den minimalen, maximalen und mittleren Spannungswert in den CIM-Daten am 09.01.2026.",
    "Wie hat sich in CIM die Wirkleistung von Load 27 am 09.01.2026 über die Zeit entwickelt?",
    "Zeig mir den historischen Verlauf der aktiven Leistung von Verbraucher 27 am 09.01.2026.",
    "Welche P-Werte hat Last 27 im Zeitverlauf vom 09.01.2026?",
    "Wie hoch ist die Blindleistung von Load 27 im Verlauf vom 09.01.2026?",
    "Zeig mir den Q-Verlauf von Verbraucher 27 am 09.01.2026.",
    "Welche reaktive Leistung hat Last 27 über die Zeit am 09.01.2026?",
    "Wie hoch war die Scheinleistung von Trafo 19-20 über die Zeit am 09.01.2026?",
    "Zeig mir die Auslastung von Transformator 19/20 im Zeitverlauf vom 09.01.2026.",
    "Wie stark war Trf 19 20 am 09.01.2026 ausgelastet?",
    "Wann hatte Trafo 19-20 am 09.01.2026 seine höchste und niedrigste Auslastung?",
    "Zu welchen Zeitpunkten war Transformator 19/20 am 09.01.26 am stärksten bzw. am schwächsten ausgelastet?",
    "Nenne mir Peak- und Minimum-Zeitpunkt der Auslastung von Trf 19 20 am 09.01.26.",
    "Wie hoch war die Wirkleistung von Load 27 am 09.01.26 zwischen 14:00 und 15:00 Uhr?",
    "Zeig mir den P-Verlauf von Verbraucher 27 im Zeitfenster 2026-01-09T14:00 bis 2026-01-09T15:00.",
    "Welche aktive Leistung hatte Last 27 am 09.01.26 zwischen 14 und 15 Uhr?",
    "Welche direkten Nachbarn hat Trafo 19-20 in den CIM-Daten? Wie hoch ist die Spannung von Trafo 19-20 am 09.01.2026?",
    "Zu welcher topologischen Komponente gehört Load 27? Wie hoch war die Wirkleistung von Load 27 am 09.01.2026?",
    "Welche Betriebsmittel sind mit Bus 19 verbunden? Wie hoch war die Spannung an Bus 19 am 09.01.2026?",
    "Wie ist die Spannung in den CIM-Daten?",
    "Wie groß ist P gerade in CIM?",
    "Wie war die durchschnittliche Auslastung von Trafo A am 09.01.2026?",
    "Wie war die Auslastung von Leitung 128 am 09.01.26?",
    "Wie groß ist die aktuelle Einspeisung von Generator 3?",
    "Wie hoch war die Lastaufnahme von Verbraucher 27?",
    "Was kam bei trafo 19-20 am 09.01.26 an wirkleistung raus?",
    "Was is die nennspannung von bus 19 aus den historischen daten?",
    "Was war los bei Verbraucher 27 am 9. Januar zwischen 14 und 15 uhr?"
]

# -----------------------------------------------------------------------------
# Resume configuration
# -----------------------------------------------------------------------------
# False = new timestamped run folder
# True  = continue an existing run folder and skip already existing JSON files
RESUME_EXISTING_RUN = False
EXISTING_OUTPUT_DIR = Path(r"C:\Users\STELLER\Documents\Masterarbeit\Test CIM\cimpy fork\cimpy\router_batch_results\20260516_223311")

OUTPUT_DIR = (
    EXISTING_OUTPUT_DIR
    if RESUME_EXISTING_RUN
    else Path("router_batch_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
)


@dataclass
class SummaryRow:
    run_id: int
    user_input: str
    route: str
    answer: str
    status: str
    error: str
    details: str
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    token_details: str
    json_path: str

    # Decision / planning trace
    domain: str
    planning_mode: str
    workflow: str
    confidence: str
    safe_to_execute: str
    num_steps: int
    steps: str
    planner_reasoning: str

    # Object / attribute decisions
    resolved_object: str
    resolved_object_confidence: str
    selected_attributes: str
    attribute_confidence: str

    # Execution trace
    num_tool_calls: int
    executed_tools: str
    failed_tool: str


def json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"


def make_json_safe(
    value: Any,
    max_depth: int = 8,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    if _seen is None:
        _seen = set()

    if _depth > max_depth:
        return "<max_depth>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    obj_id = id(value)

    if isinstance(value, (dict, list, tuple, set)):
        if obj_id in _seen:
            return "<circular_reference>"

        _seen.add(obj_id)

        if isinstance(value, dict):
            safe_dict: Dict[str, Any] = {}

            for key, item in value.items():
                key_text = str(key)

                if key_text in {
                    "services",
                    "app",
                    "project",
                    "studycase",
                    "context",
                    "topology_graph",
                    "resolved_object",
                    "snapshot_cache",
                    "network_index",
                    "cim_snapshots",
                    "base_snapshot",
                }:
                    safe_dict[key_text] = f"<omitted:{type(item).__name__}>"
                else:
                    safe_dict[key_text] = make_json_safe(
                        item,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )

            _seen.remove(obj_id)
            return safe_dict

        safe_list = [
            make_json_safe(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
                _seen=_seen,
            )
            for item in list(value)[:100]
        ]

        _seen.remove(obj_id)
        return safe_list

    return f"<non_serializable:{type(value).__name__}>"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = make_json_safe(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, indent=2, default=json_default)


def _clean_csv_cell(value: Any) -> str:
    text = _stringify(value)
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return text.strip()


def save_csv(path: Path, rows: List[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "run_id",
        "user_input",
        "route",
        "answer",
        "status",
        "error",
        "details",
        "duration_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "token_details",
        "json_path",
        "domain",
        "planning_mode",
        "workflow",
        "confidence",
        "safe_to_execute",
        "num_steps",
        "steps",
        "planner_reasoning",
        "resolved_object",
        "resolved_object_confidence",
        "selected_attributes",
        "attribute_confidence",
        "num_tool_calls",
        "executed_tools",
        "failed_tool",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for row in rows:
            raw = asdict(row)
            cleaned = {key: _clean_csv_cell(value) for key, value in raw.items()}
            writer.writerow(cleaned)

def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(make_json_safe(value), ensure_ascii=False, default=json_default)
    except Exception:
        return str(value)


def _get_nested(data: Any, path: List[str], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def extract_answer(out: Dict[str, Any]) -> str:
    if not isinstance(out, dict):
        return str(out)

    if out.get("route") == "ASK_USER":
        return str(out.get("question", ""))

    result = out.get("result", {})
    if not isinstance(result, dict):
        return ""

    candidates = [
        result.get("answer"),
        result.get("final_answer"),
        result.get("llm_answer"),
        _get_nested(result, ["summary", "answer"]),
        result.get("raw_answer"),
    ]

    messages = result.get("messages")
    if isinstance(messages, list) and messages:
        candidates.append(" ".join(str(m) for m in messages if m))

    summary_parts = result.get("summary_parts")
    if isinstance(summary_parts, list):
        part_answers = []
        for part in summary_parts:
            if isinstance(part, dict) and part.get("answer"):
                part_answers.append(str(part.get("answer")))
        if part_answers:
            candidates.append("\n\n".join(part_answers))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return ""


def extract_status(out: Dict[str, Any]) -> str:
    if not isinstance(out, dict):
        return ""
    if out.get("status"):
        return str(out.get("status"))
    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("status", ""))
    return ""


def extract_error(out: Dict[str, Any]) -> str:
    if not isinstance(out, dict):
        return ""
    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("error", ""))
    return ""


def extract_details(out: Dict[str, Any]) -> str:
    if not isinstance(out, dict):
        return ""
    result = out.get("result", {})
    if isinstance(result, dict):
        return str(result.get("details", ""))
    return ""


def _extract_plan_steps(out: Dict[str, Any]) -> tuple[int, str]:
    plan = out.get("plan", {}) if isinstance(out, dict) else {}
    if not isinstance(plan, dict):
        return 0, ""

    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        return 0, ""

    step_names: List[str] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        domain = step.get("domain", "")
        tool = step.get("tool", "")
        if domain and tool:
            step_names.append(f"{domain}.{tool}")
        elif tool:
            step_names.append(str(tool))

    return len(raw_steps), " | ".join(step_names)


def _extract_execution_trace(result: Dict[str, Any]) -> tuple[int, str, str]:
    if not isinstance(result, dict):
        return 0, "", ""

    trace = _get_nested(result, ["debug", "trace"], [])
    if not isinstance(trace, list):
        trace = result.get("debug_trace", [])
    if not isinstance(trace, list):
        return 0, "", ""

    tools: List[str] = []
    failed_tool = ""

    for item in trace:
        if not isinstance(item, dict):
            continue
        step = item.get("step", "")
        if step:
            tools.append(str(step))
        tool_result = item.get("result", {})
        if isinstance(tool_result, dict) and tool_result.get("status") == "error" and not failed_tool:
            failed_tool = str(step)

    return len(trace), " | ".join(tools), failed_tool


def _extract_pf_object_decision(result: Dict[str, Any]) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "", ""

    # PF load-change path
    resolved_load = result.get("resolved_load")
    if isinstance(resolved_load, dict):
        name = (
            resolved_load.get("loc_name")
            or resolved_load.get("name")
            or resolved_load.get("full_name")
            or resolved_load.get("asset_query")
            or ""
        )
        return str(name), _stringify(resolved_load.get("confidence", ""))

    # PF data/switch/object-resolution path
    object_resolution = _get_nested(result, ["data_query", "resolution"], None)
    if not isinstance(object_resolution, dict):
        object_resolution = _get_nested(result, ["switch", "resolution"], None)
    if not isinstance(object_resolution, dict):
        object_resolution = result.get("object_resolution")

    if isinstance(object_resolution, dict):
        selected = object_resolution.get("selected_match") or {}
        llm_decision = object_resolution.get("llm_decision") or {}
        if isinstance(selected, dict):
            name = selected.get("name") or selected.get("full_name") or object_resolution.get("asset_query") or ""
        else:
            name = object_resolution.get("asset_query") or ""
        confidence = llm_decision.get("confidence", "") if isinstance(llm_decision, dict) else ""
        return str(name), str(confidence)

    return "", ""


def _extract_cim_object_decision(result: Dict[str, Any]) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "", ""

    # Several CIM tools store resolution/debug in different places depending on workflow.
    candidates = [
        result.get("resolution"),
        result.get("equipment_resolution_debug"),
        result.get("resolved_object"),
        _get_nested(result, ["debug", "selected_equipment"], None),
    ]

    # Also inspect top-level keys from CIM final state.
    for key in ("selected_match", "equipment_resolution_debug"):
        if key in result:
            candidates.append(result.get(key))

    for candidate in candidates:
        if isinstance(candidate, dict):
            selected = candidate.get("selected_match") or candidate.get("selected_equipment") or candidate
            decision = candidate.get("llm_decision") or candidate.get("type_llm_decision") or {}
            if isinstance(selected, dict):
                name = (
                    selected.get("name")
                    or selected.get("full_name")
                    or selected.get("canonical_id")
                    or selected.get("mRID")
                    or selected.get("selected_equipment_id")
                    or ""
                )
            else:
                name = ""
            confidence = decision.get("confidence", "") if isinstance(decision, dict) else ""
            if name:
                return str(name), str(confidence)

    return "", ""


def _extract_attribute_decision(result: Dict[str, Any]) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "", ""

    # PF attribute selection
    attr_sel = _get_nested(result, ["data_query", "attribute_selection"], None)
    if isinstance(attr_sel, dict):
        instruction = attr_sel.get("instruction") if isinstance(attr_sel.get("instruction"), dict) else {}
        attrs = (
            attr_sel.get("selected_attributes")
            or attr_sel.get("selected_attribute_handles")
            or instruction.get("selected_attribute_handles")
            or instruction.get("requested_attribute_names")
            or []
        )
        decision = attr_sel.get("llm_decision") or {}
        confidence = decision.get("confidence", "") if isinstance(decision, dict) else ""
        if isinstance(attrs, list):
            return ", ".join(map(str, attrs)), str(confidence)
        return _stringify(attrs), str(confidence)

    # CIM base values / selected attributes
    for key in ("selected_attributes", "requested_base_attributes", "selected_attribute_handles"):
        attrs = result.get(key)
        if isinstance(attrs, list):
            return ", ".join(map(str, attrs)), ""

    base_debug = result.get("base_attribute_debug")
    if isinstance(base_debug, dict):
        attrs = base_debug.get("selected_attributes") or base_debug.get("requested_attributes") or []
        confidence = base_debug.get("confidence", "")
        if isinstance(attrs, list):
            return ", ".join(map(str, attrs)), str(confidence)

    return "", ""


def extract_token_usage(usage: Dict[str, Any]) -> tuple[int, int, int, str]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if not isinstance(usage, dict):
        return 0, 0, 0, ""

    for model_usage in usage.values():
        if not isinstance(model_usage, dict):
            continue

        prompt_tokens += int(model_usage.get("input_tokens", 0) or 0)
        completion_tokens += int(model_usage.get("output_tokens", 0) or 0)
        total_tokens += int(model_usage.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens, _stringify(usage)

def extract_decision_trace(out: Dict[str, Any]) -> Dict[str, Any]:
    trace = {
        "domain": "",
        "planning_mode": "",
        "workflow": "",
        "confidence": "",
        "safe_to_execute": "",
        "num_steps": 0,
        "steps": "",
        "planner_reasoning": "",
        "resolved_object": "",
        "resolved_object_confidence": "",
        "selected_attributes": "",
        "attribute_confidence": "",
        "num_tool_calls": 0,
        "executed_tools": "",
        "failed_tool": "",
    }

    if not isinstance(out, dict):
        return trace

    result = out.get("result", {})
    if not isinstance(result, dict):
        result = {}

    planner_decision = out.get("planner_decision") or result.get("planner_decision") or {}
    classification = (
        out.get("classification")
        or result.get("classification")
        or planner_decision
        or {}
    )

    if isinstance(classification, dict):
        trace["domain"] = str(
            classification.get("domain")
            or out.get("domain")
            or result.get("domain")
            or out.get("route")
            or ""
        )
        trace["planning_mode"] = str(classification.get("planning_mode", ""))
        trace["workflow"] = str(classification.get("workflow", ""))
        trace["confidence"] = str(classification.get("confidence", ""))
        trace["safe_to_execute"] = str(classification.get("safe_to_execute", ""))
        trace["planner_reasoning"] = str(
            classification.get("reasoning")
            or classification.get("rationale")
            or ""
        )

    plan = out.get("plan") or result.get("plan") or {}
    steps = []

    if isinstance(plan, dict):
        steps = plan.get("steps", []) or []
    elif isinstance(plan, list):
        steps = plan

    step_names = []
    for step in steps:
        if isinstance(step, dict):
            if "step" in step:
                step_names.append(str(step.get("step")))
            else:
                domain = step.get("domain", "")
                tool = step.get("tool", "")
                step_names.append(f"{domain}.{tool}".strip("."))

    trace["num_steps"] = len(step_names)
    trace["steps"] = " | ".join(step_names)

    debug = result.get("debug", {}) if isinstance(result, dict) else {}
    debug_trace = []

    if isinstance(debug, dict):
        debug_trace = debug.get("trace", []) or []

    if not debug_trace:
        debug_trace = result.get("debug_trace", []) or []

    executed_tools = []
    failed_tool = ""

    if isinstance(debug_trace, list):
        for item in debug_trace:
            if not isinstance(item, dict):
                continue
            step = item.get("step", "")
            if step:
                executed_tools.append(str(step))

            tool_result = item.get("result", {})
            if isinstance(tool_result, dict) and tool_result.get("status") == "error" and not failed_tool:
                failed_tool = str(step)

    trace["num_tool_calls"] = len(executed_tools)
    trace["executed_tools"] = " | ".join(executed_tools)
    trace["failed_tool"] = failed_tool

    resolved_load = result.get("resolved_load")
    if isinstance(resolved_load, dict):
        trace["resolved_object"] = str(
            resolved_load.get("loc_name")
            or resolved_load.get("name")
            or resolved_load.get("full_name")
            or ""
        )

    topology = result.get("topology", {})
    if isinstance(topology, dict) and not trace["resolved_object"]:
        resolution = topology.get("resolution", {})
        if isinstance(resolution, dict):
            selected = resolution.get("selected_match", {})
            if isinstance(selected, dict):
                trace["resolved_object"] = str(
                    selected.get("name")
                    or selected.get("full_name")
                    or selected.get("node_id")
                    or ""
                )

    switch = result.get("switch", {})
    if isinstance(switch, dict) and not trace["resolved_object"]:
        resolution = switch.get("resolution", {})
        if isinstance(resolution, dict):
            selected = resolution.get("selected_match", {})
            if isinstance(selected, dict):
                trace["resolved_object"] = str(
                    selected.get("name")
                    or selected.get("full_name")
                    or ""
                )

    data_query = result.get("data_query", {})
    if isinstance(data_query, dict):
        resolution = data_query.get("resolution", {})
        if isinstance(resolution, dict) and not trace["resolved_object"]:
            selected = resolution.get("selected_match", {})
            if isinstance(selected, dict):
                trace["resolved_object"] = str(
                    selected.get("name")
                    or selected.get("full_name")
                    or ""
                )

        attr_selection = data_query.get("attribute_selection", {})
        if isinstance(attr_selection, dict):
            attrs = (
                attr_selection.get("selected_attribute_handles")
                or attr_selection.get("selected_attributes")
                or attr_selection.get("selected_attribute_names")
                or []
            )
            if isinstance(attrs, list):
                trace["selected_attributes"] = ", ".join(map(str, attrs))

            llm_decision = attr_selection.get("llm_decision", {})
            if isinstance(llm_decision, dict):
                trace["attribute_confidence"] = str(llm_decision.get("confidence", ""))

    return trace


def _summary_row_from_payload(payload: Dict[str, Any], json_path: Path) -> SummaryRow:
    run_id = int(payload.get("run_id", 0) or 0)
    user_input = str(payload.get("user_input", ""))
    duration_seconds = float(payload.get("duration_seconds", 0.0) or 0.0)

    token_usage = payload.get("token_usage", {}) or {}
    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(token_usage.get("total_tokens", 0) or 0)
    token_details = _stringify(token_usage.get("details", {}))

    out = payload.get("router_output", {}) or {}
    if not isinstance(out, dict):
        out = {}

    decision_trace = payload.get("decision_trace") or extract_decision_trace(out)
    if not isinstance(decision_trace, dict):
        decision_trace = extract_decision_trace(out)

    route = str(out.get("route", "")) if isinstance(out, dict) else ""
    answer = extract_answer(out)
    status = extract_status(out)
    error = extract_error(out)
    details = extract_details(out)

    return SummaryRow(
        run_id=run_id,
        user_input=user_input,
        route=route,
        answer=answer,
        status=status,
        error=error,
        details=details,
        duration_seconds=duration_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        token_details=token_details,
        json_path=str(json_path),
        domain=str(decision_trace.get("domain", "")),
        planning_mode=str(decision_trace.get("planning_mode", "")),
        workflow=str(decision_trace.get("workflow", "")),
        confidence=str(decision_trace.get("confidence", "")),
        safe_to_execute=str(decision_trace.get("safe_to_execute", "")),
        num_steps=int(decision_trace.get("num_steps", 0) or 0),
        steps=str(decision_trace.get("steps", "")),
        planner_reasoning=str(decision_trace.get("planner_reasoning", "")),
        resolved_object=str(decision_trace.get("resolved_object", "")),
        resolved_object_confidence=str(decision_trace.get("resolved_object_confidence", "")),
        selected_attributes=str(decision_trace.get("selected_attributes", "")),
        attribute_confidence=str(decision_trace.get("attribute_confidence", "")),
        num_tool_calls=int(decision_trace.get("num_tool_calls", 0) or 0),
        executed_tools=str(decision_trace.get("executed_tools", "")),
        failed_tool=str(decision_trace.get("failed_tool", "")),
    )


def load_existing_summary_rows(output_dir: Path) -> tuple[List[SummaryRow], set[int]]:
    rows: List[SummaryRow] = []
    completed_run_ids: set[int] = set()

    if not output_dir.exists():
        return rows, completed_run_ids

    for json_path in sorted(output_dir.glob("*_router_result.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            row = _summary_row_from_payload(payload, json_path)
            if row.run_id <= 0:
                continue

            rows.append(row)
            completed_run_ids.add(row.run_id)
        except Exception as exc:
            print(f"WARNUNG: Vorhandene JSON konnte nicht geladen werden: {json_path} ({exc})")

    rows.sort(key=lambda row: row.run_id)
    return rows, completed_run_ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator()

    if RESUME_EXISTING_RUN:
        rows, completed_run_ids = load_existing_summary_rows(OUTPUT_DIR)
        print(f"Resume-Modus aktiv. Vorhandene Läufe: {len(completed_run_ids)}")
        if completed_run_ids:
            print(f"Fortsetzung ab erstem fehlenden Lauf nach Run {max(completed_run_ids)}.")
    else:
        rows = []
        completed_run_ids = set()

    metadata = {
        "started_at": datetime.now().isoformat(),
        "entrypoint": "cimpy.single_agent2.llm_routing.orchestrator.Orchestrator.handle",
        "equivalent_cli": r"python -m cimpy.single_agent2.llm_routing.run_router",
        "user_inputs": USER_INPUTS,
        "resume_mode": RESUME_EXISTING_RUN,
        "existing_output_dir": str(EXISTING_OUTPUT_DIR) if RESUME_EXISTING_RUN else "",
        "existing_completed_run_ids": sorted(completed_run_ids),
        "trace_columns": [
            "domain",
            "planning_mode",
            "workflow",
            "confidence",
            "safe_to_execute",
            "num_steps",
            "steps",
            "planner_reasoning",
            "resolved_object",
            "resolved_object_confidence",
            "selected_attributes",
            "attribute_confidence",
            "num_tool_calls",
            "executed_tools",
            "failed_tool",
        ],
    }
    save_json(OUTPUT_DIR / "run_metadata.json", metadata)

    total_start = time.perf_counter()

    for idx, user_input in enumerate(USER_INPUTS, start=1):
        if idx in completed_run_ids:
            print(f"\n=== RUN {idx}/{len(USER_INPUTS)} bereits vorhanden - übersprungen ===")
            continue

        print(f"\n=== RUN {idx}/{len(USER_INPUTS)} ===")
        print("INPUT:", user_input)

        output_path = OUTPUT_DIR / f"{idx:03d}_router_result.json"
        run_start = time.perf_counter()

        try:
            with get_usage_metadata_callback() as cb:
                out = orch.handle(user_input)

            duration_seconds = round(time.perf_counter() - run_start, 3)
            usage = cb.usage_metadata or {}
            prompt_tokens, completion_tokens, total_tokens, token_details = extract_token_usage(usage)
            decision_trace = extract_decision_trace(out)

            payload = {
                "run_id": idx,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "duration_seconds": duration_seconds,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "details": usage,
                },
                "decision_trace": decision_trace,
                "router_output": out,
            }
            save_json(output_path, payload)

            route = str(out.get("route", "")) if isinstance(out, dict) else ""
            answer = extract_answer(out)
            status = extract_status(out)
            error = extract_error(out)
            details = extract_details(out)

            rows.append(
                SummaryRow(
                    run_id=idx,
                    user_input=user_input,
                    route=route,
                    answer=answer,
                    status=status,
                    error=error,
                    details=details,
                    duration_seconds=duration_seconds,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    token_details=token_details,
                    json_path=str(output_path),
                    domain=decision_trace["domain"],
                    planning_mode=decision_trace["planning_mode"],
                    workflow=decision_trace["workflow"],
                    confidence=decision_trace["confidence"],
                    safe_to_execute=decision_trace["safe_to_execute"],
                    num_steps=decision_trace["num_steps"],
                    steps=decision_trace["steps"],
                    planner_reasoning=decision_trace["planner_reasoning"],
                    resolved_object=decision_trace["resolved_object"],
                    resolved_object_confidence=decision_trace["resolved_object_confidence"],
                    selected_attributes=decision_trace["selected_attributes"],
                    attribute_confidence=decision_trace["attribute_confidence"],
                    num_tool_calls=decision_trace["num_tool_calls"],
                    executed_tools=decision_trace["executed_tools"],
                    failed_tool=decision_trace["failed_tool"],
                )
            )

            print("ROUTE:", route)
            print("DOMAIN:", decision_trace["domain"])
            print("WORKFLOW:", decision_trace["workflow"])
            print("STEPS:", decision_trace["steps"])
            print("DAUER (s):", duration_seconds)
            print("TOKENS:", total_tokens)
            print("ANSWER:", answer)

        except Exception as e:
            duration_seconds = round(time.perf_counter() - run_start, 3)

            out = {
                "route": "ERROR",
                "result": {
                    "status": "error",
                    "error": type(e).__name__,
                    "details": str(e),
                    "traceback": traceback.format_exc(),
                    "answer": f"Batch-Ausführung fehlgeschlagen: {type(e).__name__}: {e}",
                },
            }
            decision_trace = extract_decision_trace(out)

            payload = {
                "run_id": idx,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "duration_seconds": duration_seconds,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "details": {},
                },
                "decision_trace": decision_trace,
                "router_output": out,
            }
            save_json(output_path, payload)

            rows.append(
                SummaryRow(
                    run_id=idx,
                    user_input=user_input,
                    route="ERROR",
                    answer=out["result"]["answer"],
                    status="error",
                    error=type(e).__name__,
                    details=str(e),
                    duration_seconds=duration_seconds,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    token_details="",
                    json_path=str(output_path),
                    domain=decision_trace["domain"],
                    planning_mode=decision_trace["planning_mode"],
                    workflow=decision_trace["workflow"],
                    confidence=decision_trace["confidence"],
                    safe_to_execute=decision_trace["safe_to_execute"],
                    num_steps=decision_trace["num_steps"],
                    steps=decision_trace["steps"],
                    planner_reasoning=decision_trace["planner_reasoning"],
                    resolved_object=decision_trace["resolved_object"],
                    resolved_object_confidence=decision_trace["resolved_object_confidence"],
                    selected_attributes=decision_trace["selected_attributes"],
                    attribute_confidence=decision_trace["attribute_confidence"],
                    num_tool_calls=decision_trace["num_tool_calls"],
                    executed_tools=decision_trace["executed_tools"],
                    failed_tool=decision_trace["failed_tool"],
                )
            )

            print("ROUTE: ERROR")
            print("DAUER (s):", duration_seconds)
            print("ERROR:", type(e).__name__, str(e))

    rows.sort(key=lambda row: row.run_id)
    save_csv(OUTPUT_DIR / "summary.csv", rows)

    total_duration_seconds = round(time.perf_counter() - total_start, 3)

    finished_metadata = {
        **metadata,
        "finished_at": datetime.now().isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "num_runs": len(rows),
        "num_existing_runs_loaded": len(completed_run_ids),
        "num_errors": sum(1 for row in rows if row.route == "ERROR" or row.status == "error"),
        "total_duration_seconds": total_duration_seconds,
        "average_duration_seconds": round(
            sum(row.duration_seconds for row in rows) / len(rows), 3
        ) if rows else 0.0,
    }
    save_json(OUTPUT_DIR / "run_metadata.json", finished_metadata)

    print("\nFertig.")
    print("Gesamtdauer (s):", total_duration_seconds)
    print("Ergebnisse in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
