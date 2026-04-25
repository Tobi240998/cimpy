from __future__ import annotations

from typing import Any, Dict, Optional


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

    def get_tool_spec(self, full_tool_name: str) -> Optional[Any]:
        domain, tool_name = self._split_name(full_tool_name)

        if domain == "cim":
            if hasattr(self.cim_registry, "get_tool_spec"):
                return self.cim_registry.get_tool_spec(tool_name)
            return None

        if domain == "pf":
            return self.pf_registry.get_tool_spec(tool_name)

        return None

    def invoke(self, full_tool_name: str, **kwargs) -> Dict[str, Any]:
        domain, tool_name = self._split_name(full_tool_name)

        if domain == "cim":
            if hasattr(self.cim_registry, "invoke"):
                return self.cim_registry.invoke(tool_name, **kwargs)

            handler = self.cim_registry._handlers.get(tool_name)
            if handler is None:
                return {
                    "status": "error",
                    "error": "tool_not_found",
                    "details": f"CIM tool not found: {tool_name}",
                }
            return handler(kwargs)

        if domain == "pf":
            return self.pf_registry.invoke(tool_name, **kwargs)

        return {
            "status": "error",
            "error": "unknown_domain",
            "details": f"Unknown domain for tool: {full_tool_name}",
        }

    def list_tool_specs(self) -> list[dict[str, Any]]:
        specs = []

        if hasattr(self.cim_registry, "list_tool_specs"):
            for spec in self.cim_registry.list_tool_specs():
                spec = dict(spec)
                spec["name"] = f"cim.{spec['name']}"
                spec["domain"] = "cim"
                specs.append(spec)

        if hasattr(self.pf_registry, "list_tool_specs"):
            for spec in self.pf_registry.list_tool_specs():
                spec = dict(spec)
                spec["name"] = f"pf.{spec['name']}"
                spec["domain"] = "powerfactory"
                specs.append(spec)

        return specs