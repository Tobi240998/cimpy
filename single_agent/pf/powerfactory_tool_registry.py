from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from cimpy.powerfactory_agent.powerfactory_mcp_tools import (
    _classify_pf_object_data_source_with_services,
    _get_load_catalog_from_services,
    _interpret_instruction_with_services,
    _resolve_load_with_services,
    _execute_change_load_with_services,
    _summarize_powerfactory_result_with_services,
    _build_topology_inventory_with_services,
    _interpret_entity_instruction_with_services,
    _resolve_entity_from_inventory_with_services,
    _interpret_switch_instruction_with_services,
    _execute_switch_operation_with_services,
    _summarize_switch_result_with_services,
    _interpret_data_query_instruction_with_services,
    _list_available_object_attributes_with_services,
    _select_pf_object_attributes_llm_with_services,
    _read_pf_object_attributes_with_services,
    _summarize_pf_object_data_result_with_services,
    _summarize_load_catalog_with_services,
    _summarize_topology_result_with_services,
    _build_unsupported_result_with_services,
    _resolve_objects_from_inventory_llm_with_services, 
    _build_unified_inventory_from_services
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
    requires_state: List[str] = field(default_factory=list)
    produces_state: List[str] = field(default_factory=list)
    is_summary: bool = False
    domain_notes: List[str] = field(default_factory=list)


class PowerFactoryToolRegistry:
    def __init__(self):
        self._registry: Dict[str, PowerFactoryToolSpec] = {
            "get_load_catalog": PowerFactoryToolSpec(
                name="get_load_catalog",
                description="Read the available catalog of PowerFactory objects of the requested type from the active project.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["services"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "get_load_catalog",
                    "requested_type": "load|transformer|line|bus|switch|generator",
                    "loads": "list[object-metadata]",
                },
                capability_tags=["powerfactory", "read", "catalog", "inventory"],
                mutating=False,
                requires_state=[],
                produces_state=["catalog_result"],
                is_summary=False,
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
                requires_state=[],
                produces_state=["instruction"],
                is_summary=False,
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
                requires_state=["instruction"],
                produces_state=["resolution"],
                is_summary=False,
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
                    "data": {"u_before": "dict[str, float]", "u_after": "dict[str, float]", "delta_u": "dict[str, float]"},
                },
                capability_tags=["powerfactory", "execution", "loadflow", "load"],
                mutating=True,
                requires_state=["instruction"],
                produces_state=["execution"],
                is_summary=False,
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
                requires_state=["execution"],
                produces_state=["summary"],
                is_summary=True,
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
                requires_state=[],
                produces_state=["graph_result"],
                is_summary=False,
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
                requires_state=["graph_result"],
                produces_state=["inventory_result"],
                is_summary=False,
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
                requires_state=["inventory_result"],
                produces_state=["entity_instruction"],
                is_summary=False,
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
                requires_state=["entity_instruction", "inventory_result", "graph_result"],
                produces_state=["entity_resolution"],
                is_summary=False,
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
                requires_state=["graph_result", "entity_resolution"],
                produces_state=["topology_result"],
                is_summary=False,
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
                requires_state=[],
                produces_state=["switch_instruction"],
                is_summary=False,
                handler=self._tool_interpret_switch_instruction,
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
                requires_state=["switch_instruction", "object_resolution"],
                produces_state=["switch_execution"],
                is_summary=False,
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
                requires_state=["switch_execution"],
                produces_state=["switch_summary", "summary"],
                is_summary=True,
                handler=self._tool_summarize_switch_result,
            ),
            "summarize_load_catalog": PowerFactoryToolSpec(
                name="summarize_load_catalog",
                description="Build a concise user-facing catalog answer.",
                input_schema={
                    "type": "object",
                    "properties": {"services": {"type": "object"}, "catalog_result": {"type": "object"}},
                    "required": ["services", "catalog_result"],
                },
                output_schema_hint={"status": "ok|error", "tool": "summarize_load_catalog", "answer": "string"},
                capability_tags=["powerfactory", "summary", "catalog"],
                mutating=False,
                requires_state=["catalog_result"],
                produces_state=["summary"],
                is_summary=True,
                handler=self._tool_summarize_load_catalog,
            ),
            "summarize_topology_result": PowerFactoryToolSpec(
                name="summarize_topology_result",
                description="Build a concise user-facing topology answer.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "topology_result": {"type": "object"},
                        "graph_result": {"type": "object"},
                        "inventory_result": {"type": "object"},
                        "entity_instruction": {"type": "object"},
                        "entity_resolution": {"type": "object"},
                    },
                    "required": ["services", "topology_result"],
                },
                output_schema_hint={"status": "ok|error", "tool": "summarize_topology_result", "answer": "string"},
                capability_tags=["powerfactory", "summary", "topology"],
                mutating=False,
                requires_state=["topology_result"],
                produces_state=["summary"],
                is_summary=True,
                handler=self._tool_summarize_topology_result,
            ),
            "unsupported_request": PowerFactoryToolSpec(
                name="unsupported_request",
                description="Return a controlled message for unsupported PowerFactory intent.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "user_input": {"type": "string"},
                        "classification": {"type": "object"},
                    },
                    "required": ["services", "user_input", "classification"],
                },
                output_schema_hint={"status": "error", "tool": "powerfactory", "answer": "string"},
                capability_tags=["powerfactory", "control"],
                mutating=False,
                requires_state=[],
                produces_state=[],
                is_summary=False,
                handler=self._tool_unsupported_request,
            ),
            "classify_data_source": PowerFactoryToolSpec(
                name="classify_data_source",
                description="Classify whether the data query should use base data or load-flow result data.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                    },
                    "required": ["services", "instruction"],
                },
                output_schema_hint={"status": "ok|error", "tool": "classify_data_source", "instruction": "dict", "effective_data_source": "base|result"},
                capability_tags=["powerfactory", "data_query", "classification", "llm"],
                mutating=False,
                requires_state=["data_query_instruction"],
                produces_state=["data_query_instruction", "data_source_decision"],
                is_summary=False,
                handler=self._tool_classify_data_source,
            ),
            "build_unified_inventory": PowerFactoryToolSpec(
                name="build_unified_inventory",
                description="Build a unified non-topology PowerFactory inventory for supported object types.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "allowed_types": {"type": "array"},
                    },
                    "required": ["services"],
                },
                output_schema_hint={"status": "ok|error", "tool": "build_unified_inventory", "inventory": "dict"},
                capability_tags=["powerfactory", "inventory", "read_only"],
                mutating=False,
                requires_state=[],
                produces_state=["unified_inventory_result", "inventory_result"],
                is_summary=False,
                handler=self._tool_build_unified_inventory,
            ),
            "interpret_data_query_instruction": PowerFactoryToolSpec(
                name="interpret_data_query_instruction",
                description="Interpret a natural-language data query into a structured PowerFactory data-query instruction.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "user_input": {"type": "string"},
                        "inventory": {"type": "object"},
                    },
                    "required": ["services", "user_input", "inventory"],
                },
                output_schema_hint={"status": "ok|error", "tool": "interpret_data_query_instruction", "instruction": "dict"},
                capability_tags=["powerfactory", "data_query", "planning", "llm"],
                mutating=False,
                requires_state=["unified_inventory_result"],
                produces_state=["data_query_instruction"],
                is_summary=False,
                handler=self._tool_interpret_data_query_instruction,
            ),
            "resolve_objects_from_inventory_llm": PowerFactoryToolSpec(
                name="resolve_objects_from_inventory_llm",
                description="Resolve PowerFactory objects by LLM-based exact candidate selection from a unified non-topology inventory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                        "inventory": {"type": "object"},
                    },
                    "required": ["services", "instruction", "inventory"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "resolve_objects_from_inventory_llm",
                    "selected_match": "dict",
                    "selected_matches": "list[dict]",
                    "selection_mode": "one|all",
                },
                capability_tags=["powerfactory", "resolution", "llm_match", "read_only"],
                mutating=False,
                requires_state=["unified_inventory_result"],
                produces_state=["object_resolution"],
                is_summary=False,
                handler=self._tool_resolve_objects_from_inventory_llm,
            ),
            "list_available_object_attributes": PowerFactoryToolSpec(
                name="list_available_object_attributes",
                description="List semantic and raw attribute options for the selected PowerFactory object.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                        "resolution": {"type": "object"},
                    },
                    "required": ["services", "instruction", "resolution"],
                },
                output_schema_hint={"status": "ok|error", "tool": "list_available_object_attributes", "attribute_options": "list"},
                capability_tags=["powerfactory", "data_query", "attributes", "read_only"],
                mutating=False,
                requires_state=["data_query_instruction", "object_resolution"],
                produces_state=["data_attribute_listing"],
                is_summary=False,
                handler=self._tool_list_available_object_attributes,
            ),
            "select_pf_object_attributes_llm": PowerFactoryToolSpec(
                name="select_pf_object_attributes_llm",
                description="Select the best matching attribute handles from the available option list.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                        "resolution": {"type": "object"},
                        "attribute_listing": {"type": "object"},
                    },
                    "required": ["services", "instruction", "resolution", "attribute_listing"],
                },
                output_schema_hint={"status": "ok|error", "tool": "select_pf_object_attributes_llm", "selected_attribute_handles": "list"},
                capability_tags=["powerfactory", "data_query", "attributes", "llm_match", "read_only"],
                mutating=False,
                requires_state=["data_query_instruction", "object_resolution", "data_attribute_listing"],
                produces_state=["data_attribute_selection"],
                is_summary=False,
                handler=self._tool_select_pf_object_attributes_llm,
            ),
            "read_pf_object_attributes": PowerFactoryToolSpec(
                name="read_pf_object_attributes",
                description="Read selected attributes from a resolved PowerFactory object with dynamic loadflow fallback.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                        "resolution": {"type": "object"},
                    },
                    "required": ["services", "instruction", "resolution"],
                },
                output_schema_hint={"status": "ok|error", "tool": "read_pf_object_attributes", "data": "dict"},
                capability_tags=["powerfactory", "data_query", "read_only"],
                mutating=False,
                requires_state=["data_query_instruction", "object_resolution", "data_attribute_selection"],
                produces_state=["data_query_execution"],
                is_summary=False,
                domain_notes=["Do not use this step before attribute selection is available."],
                handler=self._tool_read_pf_object_attributes,
            ),
            "summarize_pf_object_data_result": PowerFactoryToolSpec(
                name="summarize_pf_object_data_result",
                description="Summarize queried PowerFactory object data for the end user.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "result_payload": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["services", "result_payload", "user_input"],
                },
                output_schema_hint={"status": "ok|error", "tool": "summarize_pf_object_data_result", "answer": "string"},
                capability_tags=["powerfactory", "data_query", "summary"],
                mutating=False,
                requires_state=["data_query_execution"],
                produces_state=["data_query_summary", "summary"],
                is_summary=True,
                handler=self._tool_summarize_pf_object_data_result,
            ),
        }

    def has_tool(self, step_name: str) -> bool:
        return step_name in self._registry

    def list_tools(self) -> List[str]:
        return sorted(self._registry.keys())

    def get_tool_spec(self, step_name: str) -> PowerFactoryToolSpec | None:
        return self._registry.get(step_name)

    def get_step_contracts(self) -> Dict[str, Dict[str, Any]]:
        contracts: Dict[str, Dict[str, Any]] = {}
        for step_name, spec in self._registry.items():
            contracts[step_name] = {
                "description": spec.description,
                "requires_state": list(spec.requires_state),
                "produces_state": list(spec.produces_state),
                "mutating": spec.mutating,
                "is_summary": spec.is_summary,
                "domain_notes": list(spec.domain_notes),
            }
        return contracts


    def _tool_build_unified_inventory(self, services: Dict[str, Any], allowed_types: List[str] | None = None) -> Dict[str, Any]:
        from cimpy.powerfactory_agent.powerfactory_mcp_tools import _build_unified_inventory_from_services
        return _build_unified_inventory_from_services(services=services, allowed_types=allowed_types)
    
    def _tool_resolve_objects_from_inventory_llm(
        self,
        services: Dict[str, Any],
        instruction: Dict[str, Any],
        inventory: Dict[str, Any],
    ) -> Dict[str, Any]:
        return _resolve_objects_from_inventory_llm_with_services(
            services=services,
            instruction=instruction,
            inventory=inventory,
        )

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
                "requires_state": spec.requires_state,
                "produces_state": spec.produces_state,
                "is_summary": spec.is_summary,
                "domain_notes": spec.domain_notes,
            })
        return items

    def invoke(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = self._registry.get(name)

        if spec is None:
            return {
                "status": "error",
                "error": "tool_not_found",
                "details": f"PowerFactory tool not found: {name}",
            }

        if spec.handler is None:
            return {
                "status": "error",
                "error": "missing_handler",
                "details": f"PowerFactory tool has no handler: {name}",
            }

        return spec.handler(**payload)

    def _tool_get_load_catalog(self, services: Dict[str, Any], user_input: str, **kwargs: Any) -> Dict[str, Any]:
        return _get_load_catalog_from_services(services, user_input=user_input)

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
        inventory_result = _build_unified_inventory_from_services(
            services=services,
            allowed_types=["switch"],
        )
        if inventory_result["status"] != "ok":
            return inventory_result

        return _interpret_switch_instruction_with_services(
            services=services,
            user_input=user_input,
            inventory=inventory_result.get("inventory", {}),
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

    def _tool_summarize_load_catalog(
        self,
        services: Dict[str, Any],
        catalog_result: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _summarize_load_catalog_with_services(
            services=services,
            catalog_result=catalog_result,
        )

    def _tool_summarize_topology_result(
        self,
        services: Dict[str, Any],
        topology_result: Dict[str, Any],
        graph_result: Dict[str, Any] | None = None,
        inventory_result: Dict[str, Any] | None = None,
        entity_instruction: Dict[str, Any] | None = None,
        entity_resolution: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _summarize_topology_result_with_services(
            services=services,
            topology_result=topology_result,
            graph_result=graph_result,
            inventory_result=inventory_result,
            entity_instruction=entity_instruction,
            entity_resolution=entity_resolution,
        )

    def _tool_unsupported_request(
        self,
        services: Dict[str, Any],
        user_input: str,
        classification: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _build_unsupported_result_with_services(
            services=services,
            user_input=user_input,
            classification=classification,
        )

    def _tool_classify_data_source(
        self,
        services: Dict[str, Any],
        instruction: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _classify_pf_object_data_source_with_services(
            services=services,
            instruction=instruction,
        )


    def _tool_interpret_data_query_instruction(
        self,
        services: Dict[str, Any],
        user_input: str,
        inventory: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _interpret_data_query_instruction_with_services(
            services=services,
            user_input=user_input,
            inventory=inventory,
        )



    def _tool_list_available_object_attributes(
        self,
        services: Dict[str, Any],
        instruction: dict,
        resolution: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _list_available_object_attributes_with_services(
            services=services,
            instruction=instruction,
            resolution=resolution,
        )

    def _tool_select_pf_object_attributes_llm(
        self,
        services: Dict[str, Any],
        instruction: dict,
        resolution: dict,
        attribute_listing: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _select_pf_object_attributes_llm_with_services(
            services=services,
            instruction=instruction,
            resolution=resolution,
            attribute_listing=attribute_listing,
        )

    def _tool_read_pf_object_attributes(
        self,
        services: Dict[str, Any],
        instruction: dict,
        resolution: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _read_pf_object_attributes_with_services(
            services=services,
            instruction=instruction,
            resolution=resolution,
        )

    def _tool_summarize_pf_object_data_result(
        self,
        services: Dict[str, Any],
        result_payload: dict,
        user_input: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _summarize_pf_object_data_result_with_services(
            services=services,
            result_payload=result_payload,
            user_input=user_input,
        )