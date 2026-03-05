from __future__ import annotations

from typing import Any, Dict
from langchain_core.tools import tool

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.runner import run_historical_cim_analysis
from cimpy.powerfactory_agent.pf_runner import run_powerfactory_change


@tool("historical")
def historical_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """Führt historische CIM-Analyse aus. Erwartet mindestens user_input."""
    return run_historical_cim_analysis(user_input=user_input, cim_root=CIM_ROOT)


@tool("powerfactory")
def powerfactory_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """Führt eine PowerFactory-Änderung aus und rechnet Lastfluss (vor/nach). Erwartet mindestens user_input."""
    return run_powerfactory_change(user_input=user_input)