from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from cimpy.llm_routing.orchestrator import Orchestrator


USER_INPUTS: List[str] = [
    "Wie ist die Nennspannung von Bus 1 in PowerFactory?",
    "Gib mir die Basisdaten-Nennspannung von Bus 1 aus Powerfactory.",
    "Welche Nennspannung hat Bus 1 in Powerfactory?",
]

OUTPUT_DIR = Path("router_batch_results") / datetime.now().strftime("%Y%m%d_%H%M%S")


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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


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
        return json.dumps(value, ensure_ascii=False, default=json_default)
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
    if isinstance(result, dict):
        return str(result.get("answer", ""))
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
            or resolved_load.get("asset_query")
            or ""
        )
    elif isinstance(resolved_load, str) and resolved_load.strip():
        trace["resolved_object"] = resolved_load.strip()

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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orch = Orchestrator()
    rows: List[SummaryRow] = []

    metadata = {
        "started_at": datetime.now().isoformat(),
        "entrypoint": "cimpy.llm_routing.orchestrator.Orchestrator.handle",
        "equivalent_cli": r"python -m cimpy.llm_routing.run_router",
        "user_inputs": USER_INPUTS,
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
        print(f"\n=== RUN {idx}/{len(USER_INPUTS)} ===")
        print("INPUT:", user_input)

        output_path = OUTPUT_DIR / f"{idx:03d}_router_result.json"
        run_start = time.perf_counter()

        try:
            out = orch.handle(user_input)
            duration_seconds = round(time.perf_counter() - run_start, 3)
            decision_trace = extract_decision_trace(out)

            payload = {
                "run_id": idx,
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "duration_seconds": duration_seconds,
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

    save_csv(OUTPUT_DIR / "summary.csv", rows)

    total_duration_seconds = round(time.perf_counter() - total_start, 3)

    finished_metadata = {
        **metadata,
        "finished_at": datetime.now().isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "num_runs": len(rows),
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
