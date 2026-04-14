from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.powerfactory_domain_agent import PowerFactoryDomainAgent
from cimpy.powerfactory_agent.powerfactory_mcp_tools import (
    get_load_catalog,
    interpret_instruction,
    resolve_load,
    execute_change_load,
    summarize_powerfactory_result,
    build_data_inventory,
    interpret_data_query_instruction,
    resolve_pf_object_from_inventory_llm,
    list_available_object_attributes,
    select_pf_object_attributes_llm,
    read_pf_object_attributes,
    query_pf_object_data,
    summarize_pf_object_data_result,
)


mcp = FastMCP("powerfactory", json_response=True)


def _normalize_project_name(project_name: Optional[str]) -> str:
    return project_name or DEFAULT_PROJECT_NAME


@mcp.tool()
def run_powerfactory_agent(user_input: str, project_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the PowerFactory domain agent on a natural-language request.
    """
    agent = PowerFactoryDomainAgent(
        project_name=_normalize_project_name(project_name)
    )
    return agent.run(user_input)


@mcp.tool()
def list_powerfactory_tools(project_name: Optional[str] = None) -> Dict[str, Any]:
    """
    List the currently available PowerFactory domain tools and their metadata.
    """
    agent = PowerFactoryDomainAgent(
        project_name=_normalize_project_name(project_name)
    )
    return {
        "status": "ok",
        "tool": "list_powerfactory_tools",
        "project": _normalize_project_name(project_name),
        "available_tools": agent.get_available_tools(),
    }


@mcp.tool()
def get_powerfactory_load_catalog(project_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Return the available load catalog from the active PowerFactory project.
    """
    return get_load_catalog(
        project_name=_normalize_project_name(project_name)
    )


@mcp.tool()
def interpret_powerfactory_instruction(user_input: str, project_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Interpret a natural-language request into a structured PowerFactory instruction.
    """
    return interpret_instruction(
        user_input=user_input,
        project_name=_normalize_project_name(project_name),
    )


@mcp.tool()
def resolve_powerfactory_load(instruction: Dict[str, Any], project_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve a structured load instruction to a PowerFactory load object.
    """
    return resolve_load(
        instruction=instruction,
        project_name=_normalize_project_name(project_name),
    )


@mcp.tool()
def execute_powerfactory_change_load(instruction: Dict[str, Any], project_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute a load change in PowerFactory and run load flow before/after.
    """
    return execute_change_load(
        instruction=instruction,
        project_name=_normalize_project_name(project_name),
    )


@mcp.tool()
def summarize_powerfactory_execution(
    result_payload: Dict[str, Any],
    user_input: str,
    project_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Summarize a PowerFactory execution result.
    """
    return summarize_powerfactory_result(
        result_payload=result_payload,
        user_input=user_input,
        project_name=_normalize_project_name(project_name),
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")