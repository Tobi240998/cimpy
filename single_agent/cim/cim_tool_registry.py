from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cimpy.cimpy_time_analysis.cim_mcp_tools import (
    build_cim_services,
    _scan_snapshot_inventory_with_services,
    _resolve_cim_object_with_services,
    _list_equipment_of_type_with_services,
    _read_cim_base_values_with_services,
    _resolve_cim_comparison_with_services,
    _load_snapshot_cache_with_services,
    _query_cim_with_services,
    _compare_cim_values_with_services,
    _summarize_cim_result_with_services,
)


@dataclass(frozen=True)
class CIMToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema_hint: Dict[str, Any]
    capability_tags: List[str] = field(default_factory=list)
    mutating: bool = False
    requires_state: List[str] = field(default_factory=list)
    produces_state: List[str] = field(default_factory=list)
    is_summary: bool = False
    domain_notes: List[str] = field(default_factory=list)


class CIMToolRegistry:
    """
    CIM registry aligned structurally with the PowerFactory registry pattern:
    - registry = step contracts / source of truth / dispatch
    - cim_mcp_tools = concrete tool implementation layer

    The domain logic remains unchanged; only the layering is refactored.
    """

    def __init__(self, cim_root: str):
        self.cim_root = cim_root

        self._tool_specs: Dict[str, CIMToolSpec] = {
            "scan_snapshot_inventory": CIMToolSpec(
                name="scan_snapshot_inventory",
                description="Scan snapshot inventory and build base network index",
                input_schema={
                    "type": "object",
                    "properties": {"cim_root": {"type": "string"}},
                    "required": [],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "scan_snapshot_inventory",
                    "snapshot_inventory": "dict",
                    "base_snapshot": "dict",
                    "network_index": "dict",
                    "equipment_catalog_summary": "dict",
                },
                capability_tags=["inventory", "network_index", "equipment_catalog", "read_only"],
                mutating=False,
                requires_state=[],
                produces_state=["snapshot_inventory", "base_snapshot", "network_index", "equipment_catalog_summary"],
                is_summary=False,
                domain_notes=["Builds and merges a deterministic equipment catalog from the base snapshot."],
            ),
            "resolve_cim_object": CIMToolSpec(
                name="resolve_cim_object",
                description="Resolve asset or equipment from user query and parse time/state intent",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "user_input": {"type": "string"},
                        "network_index": {"type": "object"},
                    },
                    "required": ["user_input", "network_index"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "resolve_cim_object",
                    "parsed_query": "dict",
                    "resolved_object": "Any",
                    "resolution_mode": "string",
                    "equipment_resolution_debug": "dict",
                },
                capability_tags=["object_resolution", "query_parsing", "equipment_catalog", "llm_match", "read_only"],
                mutating=False,
                requires_state=["network_index"],
                produces_state=["parsed_query", "resolved_object", "resolution_mode", "equipment_resolution_debug"],
                is_summary=False,
                domain_notes=["Falls back to two-stage catalog resolution: equipment type first, equipment instance second."],
            ),
            "list_equipment_of_type": CIMToolSpec(
                name="list_equipment_of_type",
                description="List all concrete CIM objects of a resolved equipment type from the deterministic equipment catalog",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "user_input": {"type": "string"},
                        "network_index": {"type": "object"},
                    },
                    "required": ["user_input", "network_index"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "list_equipment_of_type",
                    "selected_type": "string",
                    "equipment_items": "list[dict]",
                    "equipment_count": "integer",
                    "answer": "string",
                },
                capability_tags=["equipment_catalog", "asset_lookup", "read_only"],
                mutating=False,
                requires_state=["network_index"],
                produces_state=["selected_type", "equipment_items", "equipment_count", "answer"],
                is_summary=False,
                domain_notes=["Uses the same type-resolution stage as equipment resolution, but returns all objects of the selected type."],
            ),

            "read_cim_base_values": CIMToolSpec(
                name="read_cim_base_values",
                description="Read static CIM base or nameplate attributes from a resolved equipment object",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "user_input": {"type": "string"},
                        "resolved_object": {"type": "object"},
                    },
                    "required": ["user_input", "resolved_object"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "read_cim_base_values",
                    "selected_attributes": "list[str]",
                    "base_values": "dict[str, Any]",
                    "answer": "string",
                },
                capability_tags=["base_values", "nameplate_data", "read_only"],
                mutating=False,
                requires_state=["resolved_object"],
                produces_state=["selected_attributes", "base_values", "answer", "base_attribute_debug"],
                is_summary=False,
                domain_notes=["Supports direct technical attribute matching and controlled semantic LLM mapping."],
            ),
            "resolve_cim_comparison": CIMToolSpec(
                name="resolve_cim_comparison",
                description="Resolve a supported CIM comparison intent into required SV metric and base attributes",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "user_input": {"type": "string"},
                        "resolved_object": {"type": "object"},
                        "parsed_query": {"type": "object"},
                    },
                    "required": ["user_input", "resolved_object"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "resolve_cim_comparison",
                    "comparison_resolution": "dict",
                    "requested_base_attributes": "list[str]",
                    "parsed_query": "dict",
                },
                capability_tags=["comparison_resolution", "read_only"],
                mutating=False,
                requires_state=["resolved_object", "parsed_query"],
                produces_state=["comparison_resolution", "requested_base_attributes", "parsed_query"],
                is_summary=False,
                domain_notes=["Determines which standard comparison should be executed before reading SV and base values."],
            ),
            "load_snapshot_cache": CIMToolSpec(
                name="load_snapshot_cache",
                description="Load relevant snapshots for the parsed time window and build snapshot cache",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "parsed_query": {"type": "object"},
                        "snapshot_inventory": {"type": "object"},
                    },
                    "required": ["parsed_query", "snapshot_inventory"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "load_snapshot_cache",
                    "cim_snapshots": "dict",
                    "required_state_types": "list[str]",
                    "snapshot_cache": "dict",
                    "snapshot_cache_summary": "dict",
                },
                capability_tags=["snapshot_loading", "state_cache", "read_only"],
                mutating=False,
                requires_state=["parsed_query", "snapshot_inventory"],
                produces_state=["cim_snapshots", "required_state_types", "snapshot_cache", "snapshot_cache_summary"],
                is_summary=False,
            ),
            "query_cim": CIMToolSpec(
                name="query_cim",
                description="Run CIM analysis query using parsed query, network index and snapshot cache",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "user_input": {"type": "string"},
                        "snapshot_cache": {"type": "object"},
                        "network_index": {"type": "object"},
                        "parsed_query": {"type": "object"},
                        "classification": {"type": "object"},
                    },
                    "required": ["user_input", "network_index"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "query_cim",
                    "answer": "string",
                },
                capability_tags=["analysis", "read_only"],
                mutating=False,
                requires_state=["network_index", "parsed_query"],
                produces_state=["answer"],
                is_summary=False,
            ),
            "compare_cim_values": CIMToolSpec(
                name="compare_cim_values",
                description="Compare resolved SV results against resolved CIM base values using a standard comparison definition",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "resolved_object": {"type": "object"},
                        "comparison_resolution": {"type": "object"},
                        "query_result_data": {"type": "object"},
                        "base_values": {"type": "object"},
                    },
                    "required": ["resolved_object", "comparison_resolution", "query_result_data", "base_values"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "compare_cim_values",
                    "comparison_result": "dict",
                    "answer": "string",
                },
                capability_tags=["comparison", "read_only"],
                mutating=False,
                requires_state=["resolved_object", "comparison_resolution", "query_result_data", "base_values"],
                produces_state=["comparison_result", "answer"],
                is_summary=False,
            ),
            "summarize_cim_result": CIMToolSpec(
                name="summarize_cim_result",
                description="Return final CIM result with debug information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cim_root": {"type": "string"},
                        "result_payload": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["user_input"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "summarize_cim_result",
                    "answer": "string",
                    "debug": "dict",
                },
                capability_tags=["result", "read_only"],
                mutating=False,
                requires_state=["answer"],
                produces_state=["summary"],
                is_summary=True,
            ),
        }

        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "scan_snapshot_inventory": self._tool_scan_snapshot_inventory,
            "resolve_cim_object": self._tool_resolve_cim_object,
            "list_equipment_of_type": self._tool_list_equipment_of_type,
            "read_cim_base_values": self._tool_read_cim_base_values,
            "resolve_cim_comparison": self._tool_resolve_cim_comparison,
            "load_snapshot_cache": self._tool_load_snapshot_cache,
            "query_cim": self._tool_query_cim,
            "compare_cim_values": self._tool_compare_cim_values,
            "summarize_cim_result": self._tool_summarize_cim_result,
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def list_tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema_hint": spec.output_schema_hint,
                "capability_tags": list(spec.capability_tags),
                "mutating": spec.mutating,
                "requires_state": list(spec.requires_state),
                "produces_state": list(spec.produces_state),
                "is_summary": spec.is_summary,
                "domain_notes": list(spec.domain_notes),
            }
            for spec in self._tool_specs.values()
        ]

    def get_step_contracts(self) -> Dict[str, Dict[str, Any]]:
        return {
            spec.name: {
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema_hint": spec.output_schema_hint,
                "capability_tags": list(spec.capability_tags),
                "mutating": spec.mutating,
                "requires_state": list(spec.requires_state),
                "produces_state": list(spec.produces_state),
                "is_summary": spec.is_summary,
                "domain_notes": list(spec.domain_notes),
            }
            for spec in self._tool_specs.values()
        }

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
    def _build_services(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return build_cim_services(cim_root=context.get("cim_root", self.cim_root))

    # ------------------------------------------------------------------
    # TOOL HANDLERS (delegate to cim_mcp_tools)
    # ------------------------------------------------------------------
    def _tool_scan_snapshot_inventory(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _scan_snapshot_inventory_with_services(services)

    def _tool_resolve_cim_object(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _resolve_cim_object_with_services(
            services=services,
            user_input=context["user_input"],
            network_index=context.get("network_index"),
            classification=context.get("classification"),
        )

    def _tool_list_equipment_of_type(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _list_equipment_of_type_with_services(
            services=services,
            user_input=context["user_input"],
            network_index=context.get("network_index"),
        )


    def _tool_read_cim_base_values(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _read_cim_base_values_with_services(
            services=services,
            user_input=context["user_input"],
            resolved_object=context.get("resolved_object") or context.get("equipment_obj"),
            parsed_query=context.get("parsed_query"),
            analysis_plan=context.get("classification"),
            requested_attributes=context.get("requested_base_attributes"),
        )

    def _tool_resolve_cim_comparison(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _resolve_cim_comparison_with_services(
            services=services,
            user_input=context["user_input"],
            resolved_object=context.get("resolved_object"),
            parsed_query=context.get("parsed_query"),
        )

    def _tool_load_snapshot_cache(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _load_snapshot_cache_with_services(
            services=services,
            parsed_query=context.get("parsed_query"),
            snapshot_inventory=context.get("snapshot_inventory"),
        )

    def _tool_query_cim(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _query_cim_with_services(
            services=services,
            user_input=context["user_input"],
            snapshot_cache=context.get("snapshot_cache", {}),
            network_index=context.get("network_index", {}),
            parsed_query=context.get("parsed_query"),
            classification=context.get("classification"),
            resolved_object=context.get("resolved_object"),
            comparison_resolution=context.get("comparison_resolution"),
        )

    def _tool_compare_cim_values(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _compare_cim_values_with_services(
            services=services,
            resolved_object=context.get("resolved_object"),
            comparison_resolution=context.get("comparison_resolution"),
            query_result_data=context.get("query_result_data"),
            base_values=context.get("base_values"),
        )

    def _tool_summarize_cim_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        services = self._build_services(context)
        if services.get("status") != "ok":
            return services
        return _summarize_cim_result_with_services(
            services=services,
            result_payload=context,
            user_input=context.get("user_input", ""),
        )
