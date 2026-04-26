from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from cimpy.single_agent2.llm_routing.unified_tool_spec import UnifiedToolSpec


@dataclass
class UnifiedToolEntry:
    spec: UnifiedToolSpec
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


class UnifiedToolRegistry:
    def __init__(self, cim_registry, pf_registry):
        self._tools: Dict[str, UnifiedToolEntry] = {}

        self._register_cim_tools(cim_registry)
        self._register_pf_tools(pf_registry)

    def _register_cim_tools(self, cim_registry) -> None:
        contracts = cim_registry.get_step_contracts()

        for tool_name, raw_spec in contracts.items():
            full_name = f"cim.{tool_name}"

            spec = self._normalize_spec(
                domain="cim",
                tool_name=tool_name,
                raw_spec=raw_spec,
            )

            def make_handler(name: str):
                def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
                    return cim_registry.invoke(name, payload)
                return handler

            self._tools[full_name] = UnifiedToolEntry(
                spec=spec,
                handler=make_handler(tool_name),
            )

    def _register_pf_tools(self, pf_registry) -> None:
        contracts = pf_registry.get_step_contracts()

        for tool_name, raw_spec in contracts.items():
            full_name = f"pf.{tool_name}"

            spec = self._normalize_spec(
                domain="pf",
                tool_name=tool_name,
                raw_spec=raw_spec,
            )

            def make_handler(name: str):
                def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
                    return pf_registry.invoke(name, payload)
                return handler

            self._tools[full_name] = UnifiedToolEntry(
                spec=spec,
                handler=make_handler(tool_name),
            )

    def _normalize_spec(
        self,
        domain: str,
        tool_name: str,
        raw_spec: Any,
    ) -> UnifiedToolSpec:
        full_name = f"{domain}.{tool_name}"

        if isinstance(raw_spec, dict):
            return UnifiedToolSpec(
                full_name=full_name,
                name=tool_name,
                domain=domain,
                description=raw_spec.get("description", ""),
                input_schema=raw_spec.get("input_schema", {}),
                output_schema_hint=raw_spec.get("output_schema_hint", {}),
                capability_tags=list(raw_spec.get("capability_tags", []) or []),
                mutating=bool(raw_spec.get("mutating", False)),
                requires_state=list(raw_spec.get("requires_state", []) or []),
                produces_state=list(raw_spec.get("produces_state", []) or []),
                is_summary=bool(raw_spec.get("is_summary", False)),
                domain_notes=list(raw_spec.get("domain_notes", []) or []),
            )

        return UnifiedToolSpec(
            full_name=full_name,
            name=tool_name,
            domain=domain,
            description=getattr(raw_spec, "description", ""),
            input_schema=getattr(raw_spec, "input_schema", {}) or {},
            output_schema_hint=getattr(raw_spec, "output_schema_hint", {}) or {},
            capability_tags=list(getattr(raw_spec, "capability_tags", []) or []),
            mutating=bool(getattr(raw_spec, "mutating", False)),
            requires_state=list(getattr(raw_spec, "requires_state", []) or []),
            produces_state=list(getattr(raw_spec, "produces_state", []) or []),
            is_summary=bool(getattr(raw_spec, "is_summary", False)),
            domain_notes=list(getattr(raw_spec, "domain_notes", []) or []),
        )

    def get_tool_spec(self, full_tool_name: str) -> Optional[UnifiedToolSpec]:
        entry = self._tools.get(full_tool_name)
        return entry.spec if entry else None

    def list_tool_specs(self) -> list[UnifiedToolSpec]:
        return [entry.spec for entry in self._tools.values()]

    def invoke(self, full_tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._tools.get(full_tool_name)

        if entry is None:
            return {
                "status": "error",
                "error": "tool_not_found",
                "details": f"Unified tool not found: {full_tool_name}",
            }

        return entry.handler(payload)