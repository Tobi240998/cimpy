from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.powerfactory_domain_agent import PowerFactoryDomainAgent
from cimpy.cimpy_time_analysis.cim_domain_agent import CIMDomainAgent

from cimpy.powerfactory_agent.powerfactory_mcp_tools import (
    get_load_catalog,
    interpret_instruction,
    resolve_load,
    execute_change_load,
    summarize_powerfactory_result,
)
from cimpy.cimpy_time_analysis.cim_mcp_tools import (
    run_cim_agent,
    list_cim_tools,
    scan_cim_snapshot_inventory,
    resolve_cim_object,
    load_cim_snapshot_cache,
    execute_cim_query,
    summarize_cim_execution,
)


mcp = FastMCP("grid", json_response=True)


def _normalize_project_name(project_name: Optional[str]) -> str:
    return project_name or DEFAULT_PROJECT_NAME


def _normalize_cim_root(cim_root: Optional[str]) -> str:
    return cim_root or CIM_ROOT


# ------------------------------------------------------------------
# DOMAIN AGENTS
# ------------------------------------------------------------------
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
def run_cim_agent_tool(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the CIM domain agent on a natural-language request.
    """
    return run_cim_agent(
        user_input=user_input,
        cim_root=_normalize_cim_root(cim_root),
    )


# ------------------------------------------------------------------
# LIST AVAILABLE TOOLS
# ------------------------------------------------------------------
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
def list_cim_tools_tool(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    List the currently available CIM domain tools and their metadata.
    """
    return list_cim_tools(
        cim_root=_normalize_cim_root(cim_root)
    )


# ------------------------------------------------------------------
# POWERFACTORY LOW-LEVEL TOOLS
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# CIM LOW-LEVEL TOOLS
# ------------------------------------------------------------------
@mcp.tool()
def scan_cim_snapshot_inventory_tool(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan the CIM snapshot inventory and build the base network index.
    """
    return scan_cim_snapshot_inventory(
        cim_root=_normalize_cim_root(cim_root)
    )


@mcp.tool()
def resolve_cim_object_tool(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve a CIM object and parse the user request.
    """
    return resolve_cim_object(
        user_input=user_input,
        cim_root=_normalize_cim_root(cim_root),
    )


@mcp.tool()
def load_cim_snapshot_cache_tool(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the CIM snapshot cache for a natural-language request.
    """
    return load_cim_snapshot_cache(
        user_input=user_input,
        cim_root=_normalize_cim_root(cim_root),
    )


@mcp.tool()
def execute_cim_query_tool(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the standard CIM registry workflow for a natural-language request.
    """
    return execute_cim_query(
        user_input=user_input,
        cim_root=_normalize_cim_root(cim_root),
    )


@mcp.tool()
def summarize_cim_execution_tool(
    result_payload: Dict[str, Any],
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Summarize a CIM execution result.
    """
    return summarize_cim_execution(
        result_payload=result_payload,
        user_input=user_input,
        cim_root=_normalize_cim_root(cim_root),
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")