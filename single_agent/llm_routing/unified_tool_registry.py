from __future__ import annotations

from typing import Any, Dict, Optional

from cimpy.single_agent.llm_routing.unified_tool_spec import UnifiedToolSpec

class UnifiedToolRegistry:
    def __init__(self, cim_registry, pf_registry):
        self.cim_registry = cim_registry
        self.pf_registry = pf_registry

    def _split_name(self, full_tool_name: str) -> tuple[str, str]:
        if "." not in full_tool_name:
            raise ValueError(f"Tool name must be prefixed, got: {full_tool_name}")

        domain, tool_name = full_tool_name.split(".", 1)

        if domain not in {"cim", "pf", "powerfactory"}:
            raise ValueError(f"Unknown tool domain: {domain}")

        if domain == "powerfactory":
            domain = "pf"

        return domain, tool_name

    def get_tool_spec(self, full_tool_name: str) -> UnifiedToolSpec | None:
        domain, tool_name = self._split_name(full_tool_name)

        raw_spec = None

        if domain == "cim":
            if hasattr(self.cim_registry, "get_tool_spec"):
                raw_spec = self.cim_registry.get_tool_spec(tool_name)
            else:
                contracts = self.cim_registry.get_step_contracts()
                raw_spec = contracts.get(tool_name)

        elif domain == "pf":
            raw_spec = self.pf_registry.get_tool_spec(tool_name)

        return self._normalize_spec(domain, tool_name, raw_spec)

    def invoke(self, full_tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        domain, tool_name = self._split_name(full_tool_name)

        if domain == "pf":
            return self.pf_registry.invoke(tool_name, payload)

        if domain == "cim":
            return self.cim_registry.invoke(tool_name, payload)


        return {
            "status": "error",
            "error": "unknown_domain",
            "details": f"Unknown domain for tool: {full_tool_name}",
        }

    def list_tool_specs(self) -> list[UnifiedToolSpec]:
        specs = []

        for domain, registry in [
            ("cim", self.cim_registry),
            ("pf", self.pf_registry),
        ]:
            if hasattr(registry, "get_step_contracts"):
                contracts = registry.get_step_contracts()
                for tool_name, raw_spec in contracts.items():
                    normalized = self._normalize_spec(domain, tool_name, raw_spec)
                    if normalized is not None:
                        specs.append(normalized)

        return specs
    
    def _normalize_spec(self, domain: str, tool_name: str, raw_spec: Any) -> UnifiedToolSpec | None:
        if raw_spec is None:
            return None

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