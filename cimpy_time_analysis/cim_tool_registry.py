from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cimpy.cimpy_time_analysis.load_cim_data import (
    scan_snapshot_inventory,
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


@dataclass(frozen=True)
class CIMToolSpec:
    name: str
    description: str
    capability_tags: List[str]
    mutating: bool = False


class CIMToolRegistry:
    """
    Minimale, lauffähige CIM Tool Registry.

    Ziel:
    - stabile interne Tools für den CIMDomainAgent
    - historisch/state-basierte Queries mit Snapshot-Cache ermöglichen
    - später 1:1 als MCP-Tools nutzbar
    """

    def __init__(self, cim_root: str):
        self.cim_root = cim_root

        self._tool_specs: Dict[str, CIMToolSpec] = {
            "scan_snapshot_inventory": CIMToolSpec(
                name="scan_snapshot_inventory",
                description="Scan snapshot inventory and build base network index",
                capability_tags=["inventory", "network_index", "read_only"],
            ),
            "resolve_cim_object": CIMToolSpec(
                name="resolve_cim_object",
                description="Resolve asset or equipment from user query and parse time/state intent",
                capability_tags=["object_resolution", "query_parsing", "read_only"],
            ),
            "load_snapshot_cache": CIMToolSpec(
                name="load_snapshot_cache",
                description="Load relevant snapshots for the parsed time window and build snapshot cache",
                capability_tags=["snapshot_loading", "state_cache", "read_only"],
            ),
            "query_cim": CIMToolSpec(
                name="query_cim",
                description="Run CIM analysis query using parsed query, network index and snapshot cache",
                capability_tags=["analysis", "read_only"],
            ),
            "summarize_cim_result": CIMToolSpec(
                name="summarize_cim_result",
                description="Return final CIM result with debug information",
                capability_tags=["result", "read_only"],
            ),
        }

        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "scan_snapshot_inventory": self._scan_snapshot_inventory,
            "resolve_cim_object": self._resolve_object,
            "load_snapshot_cache": self._load_snapshot_cache,
            "query_cim": self._query_cim,
            "summarize_cim_result": self._summarize,
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def list_tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "capability_tags": list(spec.capability_tags),
                "mutating": spec.mutating,
            }
            for spec in self._tool_specs.values()
        ]

    def get_tool_spec(self, name: str) -> Optional[CIMToolSpec]:
        return self._tool_specs.get(name)

    def invoke(self, name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._handlers.get(name)

        if not handler:
            return {
                "status": "error",
                "error": "unknown_tool",
                "tool": name,
                "answer": f"Unknown CIM tool: {name}",
            }

        try:
            return handler(context)
        except Exception as exc:
            return {
                "status": "error",
                "tool": name,
                "error": "tool_execution_failed",
                "answer": f"CIM tool '{name}' failed: {exc}",
            }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _extract_required_state_types(self, parsed_query: Dict[str, Any] | None) -> List[str]:
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

    def _should_load_states(self, parsed_query: Dict[str, Any] | None) -> bool:
        return len(self._extract_required_state_types(parsed_query)) > 0

    # ------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # ------------------------------------------------------------------
    def _scan_snapshot_inventory(self, context: Dict[str, Any]) -> Dict[str, Any]:
        inventory = scan_snapshot_inventory(self.cim_root)

        base_snapshot = load_base_snapshot(
            root_folder=self.cim_root,
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
            "snapshot_inventory": inventory,
            "base_snapshot": base_snapshot,
            "network_index": network_index,
        }

    def _resolve_object(self, context: Dict[str, Any]) -> Dict[str, Any]:
        parsed_query = interpret_user_query(
            user_input=context["user_input"],
            network_index=context.get("network_index"),
        )

        equipment_selection = parsed_query.get("equipment_selection", []) or []
        resolved = equipment_selection[0] if equipment_selection else None

        return {
            "status": "ok",
            "parsed_query": parsed_query,
            "resolved_object": resolved,
        }

    def _load_snapshot_cache(self, context: Dict[str, Any]) -> Dict[str, Any]:
        parsed_query = context.get("parsed_query") or {}
        snapshot_inventory = context.get("snapshot_inventory")

        required_state_types = self._extract_required_state_types(parsed_query)
        cim_snapshots: Dict[str, Any] = {}

        if self._should_load_states(parsed_query):
            cim_snapshots = load_snapshots_for_time_window(
                root_folder=self.cim_root,
                start_time=parsed_query.get("time_start"),
                end_time=parsed_query.get("time_end"),
                snapshot_inventory=snapshot_inventory,
            )

            if not cim_snapshots:
                cim_snapshots = load_cim_snapshots(self.cim_root)

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
            "cim_snapshots": cim_snapshots,
            "required_state_types": required_state_types,
            "snapshot_cache": snapshot_cache,
            "snapshot_cache_summary": snapshot_cache_summary,
        }

    def _query_cim(self, context: Dict[str, Any]) -> Dict[str, Any]:
        answer = handle_user_query(
            user_input=context["user_input"],
            snapshot_cache=context.get("snapshot_cache", {}),
            network_index=context.get("network_index", {}),
            parsed_query=context.get("parsed_query"),
            analysis_plan=context.get("classification"),
        )

        return {
            "status": "ok",
            "answer": answer,
        }

    def _summarize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot_inventory = context.get("snapshot_inventory") or {}
        network_index = context.get("network_index") or {}
        cim_snapshots = context.get("cim_snapshots") or {}
        snapshot_cache_summary = context.get("snapshot_cache_summary") or {}

        debug = {
            "num_inventory_snapshots": len(snapshot_inventory.get("snapshots", []))
            if snapshot_inventory else 0,
            "index_source_snapshot": network_index.get("index_source_snapshot"),
            "index_source_time_str": network_index.get("index_source_time_str"),
            "required_state_types": context.get("required_state_types", []) or [],
            "num_loaded_snapshots": len(cim_snapshots),
            "loaded_snapshot_names": list(cim_snapshots.keys()),
            "snapshot_cache_summary": snapshot_cache_summary,
            "parsed_query": context.get("parsed_query", {}) or {},
            "resolved_object": context.get("resolved_object"),
        }

        return {
            "status": "ok",
            "answer": context.get("answer", ""),
            "debug": debug,
        }