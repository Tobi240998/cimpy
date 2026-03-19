from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from cimpy.powerfactory_agent.powerfactory_mcp_tools import (
    _get_load_catalog_from_services,
    _interpret_instruction_with_services,
    _resolve_load_with_services,
    _execute_change_load_with_services,
    _summarize_powerfactory_result_with_services,
    _build_topology_inventory_with_services,
    _interpret_entity_instruction_with_services,
    _resolve_entity_from_inventory_with_services,
    _interpret_switch_instruction_with_services,
    _build_switch_inventory_from_services,
    _build_switch_inventory_payload,
    _resolve_switch_from_inventory_llm_with_services,
    _execute_switch_operation_with_services,
    _summarize_switch_result_with_services,
)
from cimpy.powerfactory_agent.powerfactory_topology_graph import (
    build_powerfactory_topology_graph_from_services,
    query_powerfactory_topology_neighbors_from_services,
)


@dataclass
class PowerFactoryToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema_hint: Dict[str, Any]
    capability_tags: List[str] = field(default_factory=list)
    mutating: bool = False
    handler: Callable[..., Dict[str, Any]] | None = None


class PowerFactoryToolRegistry:
    def __init__(self):
        self._registry: Dict[str, PowerFactoryToolSpec] = {
            "get_load_catalog": PowerFactoryToolSpec(
                name="get_load_catalog",
                description="Read the available load catalog from the active PowerFactory project.",
                input_schema={"type": "object", "properties": {"services": {"type": "object"}}, "required": ["services"]},
                output_schema_hint={"status": "ok|error", "tool": "get_load_catalog", "loads": "list[load-metadata]"},
                capability_tags=["powerfactory", "read", "catalog", "load"],
                mutating=False,
                handler=self._tool_get_load_catalog,
            ),
            "interpret_instruction": PowerFactoryToolSpec(
                name="interpret_instruction",
                description="Interpret a natural-language user request into a structured load-change instruction.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "user_input": {"type": "string"}},
                    "required": ["services", "user_input"],
                },
                output_schema_hint={"status": "ok|error", "tool": "interpret_instruction", "instruction": "dict"},
                capability_tags=["powerfactory", "planning", "nlp", "load"],
                mutating=False,
                handler=self._tool_interpret_instruction,
            ),
            "resolve_load": PowerFactoryToolSpec(
                name="resolve_load",
                description="Resolve the target load object inside the active PowerFactory project.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "instruction": {"type": "object"}},
                    "required": ["services", "instruction"],
                },
                output_schema_hint={"status": "ok|error", "tool": "resolve_load", "resolution": "dict"},
                capability_tags=["powerfactory", "resolution", "load"],
                mutating=False,
                handler=self._tool_resolve_load,
            ),
            "execute_change_load": PowerFactoryToolSpec(
                name="execute_change_load",
                description="Apply a load change in PowerFactory and run load flow before/after.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "instruction": {"type": "object"}},
                    "required": ["services", "instruction"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "execute_change_load",
                    "data": {
                        "requested_metrics": "list[str]",
                        "metric_metadata": "dict[str, dict[str, Any]]",
                        "before": "dict[str, dict[str, float]]",
                        "after": "dict[str, dict[str, float]]",
                        "delta": "dict[str, dict[str, float]]",
                        "u_before": "dict[str, float] (legacy alias when bus_voltage requested)",
                        "u_after": "dict[str, float] (legacy alias when bus_voltage requested)",
                        "delta_u": "dict[str, float] (legacy alias when bus_voltage requested)",
                    },
                },
                capability_tags=["powerfactory", "execution", "loadflow", "load"],
                mutating=True,
                handler=self._tool_execute_change_load,
            ),
            "summarize_powerfactory_result": PowerFactoryToolSpec(
                name="summarize_powerfactory_result",
                description="Summarize result data from a PowerFactory load workflow for the end user.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "result_payload": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["services", "result_payload", "user_input"],
                },
                output_schema_hint={"status": "ok|error", "tool": "summarize_powerfactory_result", "answer": "string"},
                capability_tags=["powerfactory", "summary", "result"],
                mutating=False,
                handler=self._tool_summarize_powerfactory_result,
            ),
            "build_topology_graph": PowerFactoryToolSpec(
                name="build_topology_graph",
                description="Build a topology graph from the active PowerFactory project.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "contract_cubicles": {"type": "boolean"}},
                    "required": ["services"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "build_topology_graph",
                    "graph_mode": "contracted|wiring",
                    "graph_summary": "dict",
                    "inventory": "dict",
                    "build_debug": "dict",
                },
                capability_tags=["powerfactory", "topology", "graph", "read_only"],
                mutating=False,
                handler=self._tool_build_topology_graph,
            ),
            "build_topology_inventory": PowerFactoryToolSpec(
                name="build_topology_inventory",
                description="Build a typed inventory from the current PowerFactory topology graph.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "topology_graph_result": {"type": "object"}},
                    "required": ["services", "topology_graph_result"],
                },
                output_schema_hint={"status": "ok|error", "tool": "build_topology_inventory", "inventory": "dict"},
                capability_tags=["powerfactory", "topology", "inventory", "read_only"],
                mutating=False,
                handler=self._tool_build_topology_inventory,
            ),
            "interpret_entity_instruction": PowerFactoryToolSpec(
                name="interpret_entity_instruction",
                description="Interpret a natural-language topology request into a structured generic PowerFactory entity instruction.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "user_input": {"type": "string"},
                        "inventory": {"type": "object"},
                    },
                    "required": ["services", "user_input", "inventory"],
                },
                output_schema_hint={"status": "ok|error", "tool": "interpret_entity_instruction", "instruction": "dict"},
                capability_tags=["powerfactory", "planning", "nlp", "topology", "entity"],
                mutating=False,
                handler=self._tool_interpret_entity_instruction,
            ),
            "resolve_entity_from_inventory": PowerFactoryToolSpec(
                name="resolve_entity_from_inventory",
                description="Resolve the requested generic asset from a typed topology inventory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                        "inventory": {"type": "object"},
                        "topology_graph": {"type": "object"},
                        "max_matches": {"type": "integer"},
                    },
                    "required": ["services", "instruction", "inventory", "topology_graph"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "resolve_entity_from_inventory",
                    "asset_query": "string",
                    "selected_match": "dict",
                    "matches": "list[dict]",
                },
                capability_tags=["powerfactory", "topology", "entity", "inventory", "resolution", "read_only"],
                mutating=False,
                handler=self._tool_resolve_entity_from_inventory,
            ),
            "query_topology_neighbors": PowerFactoryToolSpec(
                name="query_topology_neighbors",
                description="Query the direct neighbors of an asset in the PowerFactory topology graph.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "topology_graph": {"type": "object"},
                        "asset_query": {"type": "string"},
                        "selected_node_id": {"type": "string"},
                        "matches": {"type": "array"},
                        "max_matches": {"type": "integer"},
                    },
                    "required": ["services", "topology_graph"],
                },
                output_schema_hint={"status": "ok|error", "tool": "query_topology_neighbors", "selected_node": "dict", "neighbors": "list[dict]"},
                capability_tags=["powerfactory", "topology", "neighbors", "read_only"],
                mutating=False,
                handler=self._tool_query_topology_neighbors,
            ),
            "interpret_switch_instruction": PowerFactoryToolSpec(
                name="interpret_switch_instruction",
                description="Interpret a natural-language switch request into a structured switch operation instruction.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "user_input": {"type": "string"}},
                    "required": ["services", "user_input"],
                },
                output_schema_hint={"status": "ok|error", "tool": "interpret_switch_instruction", "instruction": "dict"},
                capability_tags=["powerfactory", "planning", "nlp", "switch"],
                mutating=False,
                handler=self._tool_interpret_switch_instruction,
            ),
            "resolve_switch_from_inventory_llm": PowerFactoryToolSpec(
                name="resolve_switch_from_inventory_llm",
                description="Resolve a switch by LLM-based selection from the exact available switch candidate list.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "instruction": {"type": "object"}},
                    "required": ["services", "instruction"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "resolve_switch_from_inventory_llm",
                    "asset_query": "string",
                    "selected_match": "dict",
                    "llm_decision": "dict",
                },
                capability_tags=["powerfactory", "switch", "llm_match", "resolution", "read_only"],
                mutating=False,
                handler=self._tool_resolve_switch_from_inventory_llm,
            ),
            "execute_switch_operation": PowerFactoryToolSpec(
                name="execute_switch_operation",
                description="Execute an open/close/toggle operation on a resolved switch object in PowerFactory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                        "resolution": {"type": "object"},
                        "run_loadflow_after": {"type": "boolean"},
                    },
                    "required": ["services", "instruction", "resolution"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "execute_switch_operation",
                    "switch": "dict",
                    "state_before": "string|None",
                    "state_after": "string|None",
                    "loadflow": "dict",
                },
                capability_tags=["powerfactory", "execution", "switch", "topology"],
                mutating=True,
                handler=self._tool_execute_switch_operation,
            ),
            "summarize_switch_result": PowerFactoryToolSpec(
                name="summarize_switch_result",
                description="Summarize the result of a switch state change for the end user.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "result_payload": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["services", "result_payload", "user_input"],
                },
                output_schema_hint={"status": "ok|error", "tool": "summarize_switch_result", "answer": "string"},
                capability_tags=["powerfactory", "summary", "switch"],
                mutating=False,
                handler=self._tool_summarize_switch_result,
            ),
        }

    def has_tool(self, step_name: str) -> bool:
        return step_name in self._registry

    def list_tools(self) -> List[str]:
        return sorted(self._registry.keys())

    def get_tool_spec(self, step_name: str) -> PowerFactoryToolSpec | None:
        return self._registry.get(step_name)

    def list_tool_specs(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for name in self.list_tools():
            spec = self._registry[name]
            items.append({
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema_hint": spec.output_schema_hint,
                "capability_tags": spec.capability_tags,
                "mutating": spec.mutating,
            })
        return items

    def invoke(self, step_name: str, **kwargs: Any) -> Dict[str, Any]:
        spec = self.get_tool_spec(step_name)

        if spec is None or spec.handler is None:
            return {
                "status": "error",
                "tool": "powerfactory_tool_registry",
                "error": f"Unknown tool step: {step_name}",
                "available_tools": self.list_tools(),
            }

        return spec.handler(**kwargs)

    def _tool_get_load_catalog(self, services: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return _get_load_catalog_from_services(services)

    def _tool_interpret_instruction(self, services: Dict[str, Any], user_input: str, **kwargs: Any) -> Dict[str, Any]:
        return _interpret_instruction_with_services(services=services, user_input=user_input)

    def _tool_resolve_load(self, services: Dict[str, Any], instruction: dict, **kwargs: Any) -> Dict[str, Any]:
        return _resolve_load_with_services(services=services, instruction=instruction)

    def _tool_execute_change_load(self, services: Dict[str, Any], instruction: dict, **kwargs: Any) -> Dict[str, Any]:
        return _execute_change_load_with_services(services=services, instruction=instruction)

    def _tool_summarize_powerfactory_result(
        self,
        services: Dict[str, Any],
        result_payload: dict,
        user_input: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _summarize_powerfactory_result_with_services(
            services=services,
            result_payload=result_payload,
            user_input=user_input,
        )

    def _tool_build_topology_graph(
        self,
        services: Dict[str, Any],
        contract_cubicles: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return build_powerfactory_topology_graph_from_services(
            services=services,
            contract_cubicles=contract_cubicles,
        )

    def _tool_build_topology_inventory(
        self,
        services: Dict[str, Any],
        topology_graph_result: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _build_topology_inventory_with_services(
            services=services,
            topology_graph_result=topology_graph_result,
        )

    def _tool_interpret_entity_instruction(
        self,
        services: Dict[str, Any],
        user_input: str,
        inventory: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _interpret_entity_instruction_with_services(
            services=services,
            user_input=user_input,
            inventory=inventory,
        )

    def _tool_resolve_entity_from_inventory(
        self,
        services: Dict[str, Any],
        instruction: dict,
        inventory: Dict[str, Any],
        topology_graph: Any,
        max_matches: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _resolve_entity_from_inventory_with_services(
            services=services,
            instruction=instruction,
            inventory=inventory,
            topology_graph=topology_graph,
            max_matches=max_matches,
        )

    def _tool_query_topology_neighbors(
        self,
        services: Dict[str, Any],
        topology_graph: Any,
        asset_query: str | None = None,
        selected_node_id: str | None = None,
        matches: list | None = None,
        max_matches: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return query_powerfactory_topology_neighbors_from_services(
            services=services,
            topology_graph=topology_graph,
            asset_query=asset_query or "",
            selected_node_id=selected_node_id,
            matches=matches or [],
            max_matches=max_matches,
        )

    def _tool_interpret_switch_instruction(
        self,
        services: Dict[str, Any],
        user_input: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        switch_inventory_result = _build_switch_inventory_from_services(services)
        if switch_inventory_result["status"] != "ok":
            return switch_inventory_result

        inventory = _build_switch_inventory_payload(switch_inventory_result.get("switches", []))
        return _interpret_switch_instruction_with_services(
            services=services,
            user_input=user_input,
            inventory=inventory,
        )

    def _tool_resolve_switch_from_inventory_llm(
        self,
        services: Dict[str, Any],
        instruction: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        switch_inventory_result = _build_switch_inventory_from_services(services)
        if switch_inventory_result["status"] != "ok":
            return switch_inventory_result

        inventory = _build_switch_inventory_payload(switch_inventory_result.get("switches", []))
        return _resolve_switch_from_inventory_llm_with_services(
            services=services,
            instruction=instruction,
            inventory=inventory,
        )

    def _tool_execute_switch_operation(
        self,
        services: Dict[str, Any],
        instruction: dict,
        resolution: dict,
        run_loadflow_after: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _execute_switch_operation_with_services(
            services=services,
            instruction=instruction,
            resolution=resolution,
            run_loadflow_after=run_loadflow_after,
        )

    def _tool_summarize_switch_result(
        self,
        services: Dict[str, Any],
        result_payload: dict,
        user_input: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _summarize_switch_result_with_services(
            services=services,
            result_payload=result_payload,
            user_input=user_input,
        )