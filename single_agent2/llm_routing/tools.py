from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import tool

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.cim_domain_agent import CIMDomainAgent

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.powerfactory_domain_agent import PowerFactoryDomainAgent


# ------------------------------------------------------------------
# HISTORICAL TOOL (DOMAIN AGENT)
# ------------------------------------------------------------------
@tool("historical")
def historical_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Führt eine CIM-/Historical-Operation über den CIMDomainAgent aus.
    Der Router bleibt unverändert und ruft weiterhin nur das Tool "historical" auf.
    """

    cim_root = (
        kwargs.get("cim_root")
        or kwargs.get("root_folder")
        or CIM_ROOT
    )

    agent = CIMDomainAgent(cim_root=cim_root)
    return agent.run(user_input)


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