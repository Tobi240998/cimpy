from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from cimpy.powerfactory_agent.powerfactory_mcp_tools import (
    _get_load_catalog_from_services,
    _interpret_instruction_with_services,
    _resolve_load_with_services,
    _execute_change_load_with_services,
    _summarize_powerfactory_result_with_services,
)


@dataclass
class PowerFactoryToolSpec:
    """
    MCP-near internal tool specification.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema_hint: Dict[str, Any]
    capability_tags: List[str] = field(default_factory=list)
    mutating: bool = False
    handler: Callable[..., Dict[str, Any]] | None = None


class PowerFactoryToolRegistry:
    """
    Registry for PowerFactory domain tools.
    Maps planned step names to executable internal tool functions
    and exposes MCP-near tool metadata.
    """

    def __init__(self):
        self._registry: Dict[str, PowerFactoryToolSpec] = {
            "get_load_catalog": PowerFactoryToolSpec(
                name="get_load_catalog",
                description="Read the available load catalog from the active PowerFactory project.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                    },
                    "required": ["services"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "get_load_catalog",
                    "loads": "list[load-metadata]",
                },
                capability_tags=["powerfactory", "read", "catalog", "load"],
                mutating=False,
                handler=self._tool_get_load_catalog,
            ),
            "interpret_instruction": PowerFactoryToolSpec(
                name="interpret_instruction",
                description="Interpret a natural-language user request into a structured load-change instruction.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["services", "user_input"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "interpret_instruction",
                    "instruction": "dict",
                },
                capability_tags=["powerfactory", "planning", "nlp", "load"],
                mutating=False,
                handler=self._tool_interpret_instruction,
            ),
            "resolve_load": PowerFactoryToolSpec(
                name="resolve_load",
                description="Resolve the target load object inside the active PowerFactory project.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                    },
                    "required": ["services", "instruction"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "resolve_load",
                    "resolution": "dict",
                },
                capability_tags=["powerfactory", "resolution", "load"],
                mutating=False,
                handler=self._tool_resolve_load,
            ),
            "execute_change_load": PowerFactoryToolSpec(
                name="execute_change_load",
                description="Apply a load change in PowerFactory and run load flow before/after.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "instruction": {"type": "object"},
                    },
                    "required": ["services", "instruction"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "execute_change_load",
                    "data": {
                        "u_before": "dict[str, float]",
                        "u_after": "dict[str, float]",
                        "delta_u": "dict[str, float]",
                    },
                },
                capability_tags=["powerfactory", "execution", "loadflow", "load"],
                mutating=True,
                handler=self._tool_execute_change_load,
            ),
            "summarize_powerfactory_result": PowerFactoryToolSpec(
                name="summarize_powerfactory_result",
                description="Summarize result data from a PowerFactory workflow for the end user.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "services": {"type": "object"},
                        "result_payload": {"type": "object"},
                        "user_input": {"type": "string"},
                    },
                    "required": ["services", "result_payload", "user_input"],
                },
                output_schema_hint={
                    "status": "ok|error",
                    "tool": "summarize_powerfactory_result",
                    "answer": "string",
                },
                capability_tags=["powerfactory", "summary", "result"],
                mutating=False,
                handler=self._tool_summarize_powerfactory_result,
            ),
        }

    # ------------------------------------------------------------------
    # REGISTRY ACCESS
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TOOL WRAPPERS
    # ------------------------------------------------------------------
    def _tool_get_load_catalog(self, services: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return _get_load_catalog_from_services(services)

    def _tool_interpret_instruction(
        self,
        services: Dict[str, Any],
        user_input: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _interpret_instruction_with_services(
            services=services,
            user_input=user_input,
        )

    def _tool_resolve_load(
        self,
        services: Dict[str, Any],
        instruction: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _resolve_load_with_services(
            services=services,
            instruction=instruction,
        )

    def _tool_execute_change_load(
        self,
        services: Dict[str, Any],
        instruction: dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _execute_change_load_with_services(
            services=services,
            instruction=instruction,
        )

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