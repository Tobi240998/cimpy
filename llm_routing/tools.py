from __future__ import annotations

from typing import Any, Dict
from langchain_core.tools import tool

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.runner import run_historical_cim_analysis

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.powerfactory_domain_agent import PowerFactoryDomainAgent


# ------------------------------------------------------------------
# HISTORICAL TOOL
# ------------------------------------------------------------------
@tool("historical")
def historical_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """Führt historische CIM-Analyse aus."""
    return run_historical_cim_analysis(
        user_input=user_input,
        cim_root=CIM_ROOT,
    )


# ------------------------------------------------------------------
# POWERFACTORY TOOL (DOMAIN AGENT)
# ------------------------------------------------------------------
@tool("powerfactory")
def powerfactory_tool(
    user_input: str,
    project: str | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Führt eine PowerFactory-Operation über den PowerFactoryDomainAgent aus.
    """

    project_name = (
        project
        or kwargs.get("project")
        or kwargs.get("project_name")
        or DEFAULT_PROJECT_NAME
    )

    agent = PowerFactoryDomainAgent(project_name=project_name)

    return agent.run(user_input)