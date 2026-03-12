from __future__ import annotations

from typing import Any, Dict
from langchain_core.tools import tool

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.runner import run_historical_cim_analysis
from cimpy.powerfactory_agent.powerfactory_mcp_tools import run_powerfactory_pipeline


@tool("historical")
def historical_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """Führt historische CIM-Analyse aus. Erwartet mindestens user_input."""
    return run_historical_cim_analysis(user_input=user_input, cim_root=CIM_ROOT)


@tool("powerfactory")
def powerfactory_tool(user_input: str, project: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Führt eine PowerFactory-Änderung über den MCP-fähigen Tool-Layer aus."""
    project_name = project or kwargs.get("project") or kwargs.get("project_name")

    if project_name:
        return run_powerfactory_pipeline(
            user_input=user_input,
            project_name=project_name,
        )

    return run_powerfactory_pipeline(
        user_input=user_input,
    )