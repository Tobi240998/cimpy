from __future__ import annotations

from typing import Any, Dict, Optional

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.cim_domain_agent import CIMDomainAgent
from cimpy.cimpy_time_analysis.cim_tool_registry import CIMToolRegistry


def _normalize_cim_root(cim_root: Optional[str] = None) -> str:
    return cim_root or CIM_ROOT


def _build_registry(cim_root: Optional[str] = None) -> CIMToolRegistry:
    return CIMToolRegistry(cim_root=_normalize_cim_root(cim_root))


# ------------------------------------------------------------------
# DOMAIN AGENT
# ------------------------------------------------------------------
def run_cim_agent(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the CIM domain agent on a natural-language request.
    """
    agent = CIMDomainAgent(cim_root=_normalize_cim_root(cim_root))
    return agent.run(user_input)


def list_cim_tools(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    List the currently available CIM domain tools and their metadata.
    """
    registry = _build_registry(cim_root)
    return {
        "status": "ok",
        "tool": "list_cim_tools",
        "cim_root": _normalize_cim_root(cim_root),
        "available_tools": registry.list_tool_specs(),
    }


# ------------------------------------------------------------------
# LOW-LEVEL REGISTRY TOOLS
# ------------------------------------------------------------------
def scan_cim_snapshot_inventory(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan the CIM snapshot inventory and build the base network index.
    """
    registry = _build_registry(cim_root)
    return registry.invoke(
        "scan_snapshot_inventory",
        context={
            "cim_root": _normalize_cim_root(cim_root),
        },
    )


def resolve_cim_object(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve the CIM object and parse the user request.
    """
    registry = _build_registry(cim_root)

    scan_result = registry.invoke(
        "scan_snapshot_inventory",
        context={
            "cim_root": _normalize_cim_root(cim_root),
        },
    )
    if scan_result.get("status") != "ok":
        return scan_result

    context = {
        "user_input": user_input,
        "cim_root": _normalize_cim_root(cim_root),
        **scan_result,
    }
    return registry.invoke("resolve_cim_object", context=context)


def load_cim_snapshot_cache(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve the request and load the relevant CIM snapshot cache.
    """
    registry = _build_registry(cim_root)

    scan_result = registry.invoke(
        "scan_snapshot_inventory",
        context={
            "cim_root": _normalize_cim_root(cim_root),
        },
    )
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = registry.invoke(
        "resolve_cim_object",
        context={
            "user_input": user_input,
            "cim_root": _normalize_cim_root(cim_root),
            **scan_result,
        },
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    context = {
        "user_input": user_input,
        "cim_root": _normalize_cim_root(cim_root),
        **scan_result,
        **resolve_result,
    }
    return registry.invoke("load_snapshot_cache", context=context)


def execute_cim_query(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a CIM query using the standard registry flow.
    """
    registry = _build_registry(cim_root)

    scan_result = registry.invoke(
        "scan_snapshot_inventory",
        context={
            "cim_root": _normalize_cim_root(cim_root),
        },
    )
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = registry.invoke(
        "resolve_cim_object",
        context={
            "user_input": user_input,
            "cim_root": _normalize_cim_root(cim_root),
            **scan_result,
        },
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    cache_result = registry.invoke(
        "load_snapshot_cache",
        context={
            "user_input": user_input,
            "cim_root": _normalize_cim_root(cim_root),
            **scan_result,
            **resolve_result,
        },
    )
    if cache_result.get("status") != "ok":
        return cache_result

    query_result = registry.invoke(
        "query_cim",
        context={
            "user_input": user_input,
            "cim_root": _normalize_cim_root(cim_root),
            **scan_result,
            **resolve_result,
            **cache_result,
        },
    )
    if query_result.get("status") != "ok":
        return query_result

    summary_result = registry.invoke(
        "summarize_cim_result",
        context={
            "user_input": user_input,
            "cim_root": _normalize_cim_root(cim_root),
            **scan_result,
            **resolve_result,
            **cache_result,
            **query_result,
        },
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
    """
    Summarize a CIM execution result.
    """
    registry = _build_registry(cim_root)

    context = {
        "user_input": user_input,
        "cim_root": _normalize_cim_root(cim_root),
        **(result_payload or {}),
    }
    return registry.invoke("summarize_cim_result", context=context)