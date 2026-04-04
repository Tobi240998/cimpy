from __future__ import annotations

from typing import Any, Dict, List, Optional

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.load_cim_data import (
    scan_snapshot_inventory as _scan_snapshot_inventory_raw,
    load_base_snapshot,
    build_network_index_from_snapshot,
    load_snapshots_for_time_window,
    load_cim_snapshots,
)
from cimpy.cimpy_time_analysis.cim_snapshot_cache import (
    preprocess_snapshots_for_states,
    summarize_snapshot_cache,
)
from cimpy.cimpy_time_analysis.llm_object_mapping import interpret_user_query
from cimpy.cimpy_time_analysis.llm_cim_orchestrator import handle_user_query


def _normalize_cim_root(cim_root: Optional[str] = None) -> str:
    return cim_root or CIM_ROOT


# ------------------------------------------------------------------
# CONTEXT / SERVICES
# ------------------------------------------------------------------
def build_cim_services(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a minimal CIM service/context payload.

    Kept intentionally lightweight so the current logic stays unchanged while
    the module becomes structurally closer to the PowerFactory MCP layer.
    """
    normalized_root = _normalize_cim_root(cim_root)
    return {
        "status": "ok",
        "tool": "cim_context",
        "cim_root": normalized_root,
    }


# ------------------------------------------------------------------
# INTERNAL HELPERS (moved from registry, logic unchanged)
# ------------------------------------------------------------------
def _extract_required_state_types(parsed_query: Dict[str, Any] | None) -> List[str]:
    parsed_query = parsed_query or {}
    state_detected = parsed_query.get("state_detected", []) or []

    state_types: List[str] = []

    if "SvVoltage" in state_detected:
        state_types.append("SvVoltage")
    if "SvPowerFlow" in state_detected:
        state_types.append("SvPowerFlow")

    out: List[str] = []
    seen = set()

    for state_type in state_types:
        if state_type not in seen:
            out.append(state_type)
            seen.add(state_type)

    return out


def _should_load_states(parsed_query: Dict[str, Any] | None) -> bool:
    return len(_extract_required_state_types(parsed_query)) > 0


# ------------------------------------------------------------------
# LOW-LEVEL TOOL IMPLEMENTATIONS
# ------------------------------------------------------------------
def _scan_snapshot_inventory_with_services(
    services: Dict[str, Any],
) -> Dict[str, Any]:
    cim_root = services["cim_root"]

    inventory = _scan_snapshot_inventory_raw(cim_root)

    base_snapshot = load_base_snapshot(
        root_folder=cim_root,
        snapshot_inventory=inventory,
    )

    network_index = build_network_index_from_snapshot(base_snapshot)

    if not network_index or not network_index.get("equipment_name_index"):
        return {
            "status": "error",
            "tool": "scan_snapshot_inventory",
            "error": "invalid_network_index",
            "answer": "Es konnte kein gültiger Netzwerkindex aus dem Basissnapshot aufgebaut werden.",
        }

    return {
        "status": "ok",
        "tool": "scan_snapshot_inventory",
        "cim_root": cim_root,
        "snapshot_inventory": inventory,
        "base_snapshot": base_snapshot,
        "network_index": network_index,
    }


def _resolve_cim_object_with_services(
    services: Dict[str, Any],
    user_input: str,
    network_index: Dict[str, Any] | None,
) -> Dict[str, Any]:
    parsed_query = interpret_user_query(
        user_input=user_input,
        network_index=network_index,
    )

    equipment_selection = parsed_query.get("equipment_selection", []) or []
    resolved = equipment_selection[0] if equipment_selection else None

    return {
        "status": "ok",
        "tool": "resolve_cim_object",
        "cim_root": services["cim_root"],
        "user_input": user_input,
        "parsed_query": parsed_query,
        "resolved_object": resolved,
    }


def _load_snapshot_cache_with_services(
    services: Dict[str, Any],
    parsed_query: Dict[str, Any] | None,
    snapshot_inventory: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cim_root = services["cim_root"]
    parsed_query = parsed_query or {}
    snapshot_inventory = snapshot_inventory or {}

    required_state_types = _extract_required_state_types(parsed_query)
    cim_snapshots: Dict[str, Any] = {}

    if _should_load_states(parsed_query):
        cim_snapshots = load_snapshots_for_time_window(
            root_folder=cim_root,
            start_time=parsed_query.get("time_start"),
            end_time=parsed_query.get("time_end"),
            snapshot_inventory=snapshot_inventory,
        )

        if not cim_snapshots:
            cim_snapshots = load_cim_snapshots(cim_root)

    if cim_snapshots and required_state_types:
        snapshot_cache = preprocess_snapshots_for_states(
            cim_snapshots=cim_snapshots,
            state_types=required_state_types,
        )
    else:
        snapshot_cache = {}

    snapshot_cache_summary = summarize_snapshot_cache(snapshot_cache)

    return {
        "status": "ok",
        "tool": "load_snapshot_cache",
        "cim_root": cim_root,
        "cim_snapshots": cim_snapshots,
        "required_state_types": required_state_types,
        "snapshot_cache": snapshot_cache,
        "snapshot_cache_summary": snapshot_cache_summary,
    }


def _query_cim_with_services(
    services: Dict[str, Any],
    user_input: str,
    snapshot_cache: Dict[str, Any] | None,
    network_index: Dict[str, Any] | None,
    parsed_query: Dict[str, Any] | None,
    classification: Dict[str, Any] | None,
) -> Dict[str, Any]:
    answer = handle_user_query(
        user_input=user_input,
        snapshot_cache=snapshot_cache or {},
        network_index=network_index or {},
        parsed_query=parsed_query,
        analysis_plan=classification,
    )

    return {
        "status": "ok",
        "tool": "query_cim",
        "cim_root": services["cim_root"],
        "answer": answer,
    }


def _summarize_cim_result_with_services(
    services: Dict[str, Any],
    result_payload: Dict[str, Any],
    user_input: str,
) -> Dict[str, Any]:
    snapshot_inventory = result_payload.get("snapshot_inventory") or {}
    network_index = result_payload.get("network_index") or {}
    cim_snapshots = result_payload.get("cim_snapshots") or {}
    snapshot_cache_summary = result_payload.get("snapshot_cache_summary") or {}

    debug = {
        "num_inventory_snapshots": len(snapshot_inventory.get("snapshots", []))
        if snapshot_inventory else 0,
        "index_source_snapshot": network_index.get("index_source_snapshot"),
        "index_source_time_str": network_index.get("index_source_time_str"),
        "required_state_types": result_payload.get("required_state_types", []) or [],
        "num_loaded_snapshots": len(cim_snapshots),
        "loaded_snapshot_names": list(cim_snapshots.keys()),
        "snapshot_cache_summary": snapshot_cache_summary,
        "parsed_query": result_payload.get("parsed_query", {}) or {},
        "resolved_object": result_payload.get("resolved_object"),
    }

    return {
        "status": "ok",
        "tool": "summarize_cim_result",
        "cim_root": services["cim_root"],
        "user_input": user_input,
        "answer": result_payload.get("answer", ""),
        "debug": debug,
    }


# ------------------------------------------------------------------
# PUBLIC MCP-STYLE TOOLS (atomic)
# ------------------------------------------------------------------
def scan_snapshot_inventory(cim_root: Optional[str] = None) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services
    return _scan_snapshot_inventory_with_services(services)


def resolve_cim_object(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    return _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )


def load_snapshot_cache(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    return _load_snapshot_cache_with_services(
        services=services,
        parsed_query=resolve_result.get("parsed_query"),
        snapshot_inventory=scan_result.get("snapshot_inventory"),
    )


def query_cim(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    cache_result = _load_snapshot_cache_with_services(
        services=services,
        parsed_query=resolve_result.get("parsed_query"),
        snapshot_inventory=scan_result.get("snapshot_inventory"),
    )
    if cache_result.get("status") != "ok":
        return cache_result

    return _query_cim_with_services(
        services=services,
        user_input=user_input,
        snapshot_cache=cache_result.get("snapshot_cache"),
        network_index=scan_result.get("network_index"),
        parsed_query=resolve_result.get("parsed_query"),
        classification=None,
    )


def summarize_cim_result(
    result_payload: Dict[str, Any],
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    return _summarize_cim_result_with_services(
        services=services,
        result_payload=result_payload or {},
        user_input=user_input,
    )


# ------------------------------------------------------------------
# DOMAIN AGENT / REGISTRY HELPERS
# ------------------------------------------------------------------
def run_cim_agent(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the CIM domain agent on a natural-language request.
    Local import avoids a circular dependency with the registry.
    """
    from cimpy.cimpy_time_analysis.cim_domain_agent import CIMDomainAgent

    agent = CIMDomainAgent(cim_root=_normalize_cim_root(cim_root))
    return agent.run(user_input)


def list_cim_tools(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    List the currently available CIM domain tools and their metadata.
    Local import avoids a circular dependency with the registry.
    """
    from cimpy.cimpy_time_analysis.cim_tool_registry import CIMToolRegistry

    registry = CIMToolRegistry(cim_root=_normalize_cim_root(cim_root))
    return {
        "status": "ok",
        "tool": "list_cim_tools",
        "cim_root": _normalize_cim_root(cim_root),
        "available_tools": registry.list_tool_specs(),
    }


# ------------------------------------------------------------------
# BACKWARD-COMPATIBLE ALIASES
# ------------------------------------------------------------------
def scan_cim_snapshot_inventory(cim_root: Optional[str] = None) -> Dict[str, Any]:
    return scan_snapshot_inventory(cim_root=cim_root)


def load_cim_snapshot_cache(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    return load_snapshot_cache(user_input=user_input, cim_root=cim_root)


def execute_cim_query(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    cache_result = _load_snapshot_cache_with_services(
        services=services,
        parsed_query=resolve_result.get("parsed_query"),
        snapshot_inventory=scan_result.get("snapshot_inventory"),
    )
    if cache_result.get("status") != "ok":
        return cache_result

    query_result = _query_cim_with_services(
        services=services,
        user_input=user_input,
        snapshot_cache=cache_result.get("snapshot_cache"),
        network_index=scan_result.get("network_index"),
        parsed_query=resolve_result.get("parsed_query"),
        classification=None,
    )
    if query_result.get("status") != "ok":
        return query_result

    summary_result = _summarize_cim_result_with_services(
        services=services,
        result_payload={
            **scan_result,
            **resolve_result,
            **cache_result,
            **query_result,
        },
        user_input=user_input,
    )
    if summary_result.get("status") != "ok":
        return summary_result

    return {
        "status": "ok",
        "tool": "execute_cim_query",
        "cim_root": _normalize_cim_root(cim_root),
        "answer": summary_result.get("answer", ""),
        "debug": {
            "scan_snapshot_inventory": scan_result,
            "resolve_cim_object": resolve_result,
            "load_snapshot_cache": cache_result,
            "query_cim": query_result,
            "summarize_cim_result": summary_result,
        },
    }


def summarize_cim_execution(
    result_payload: Dict[str, Any],
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    return summarize_cim_result(
        result_payload=result_payload,
        user_input=user_input,
        cim_root=cim_root,
    )
