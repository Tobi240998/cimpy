from __future__ import annotations

from typing import Any, Dict
from langchain_core.tools import tool

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.runner import run_historical_cim_analysis


@tool("historical", return_direct=False)
def historical_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Führt historische CIM-Analyse aus.
    Erwartet mindestens user_input. Weitere kwargs (equipment_id, time_range, metric, ...) sind optional.
    """
    # Du kannst kwargs später in run_historical_cim_analysis einspeisen, wenn du dein Backend darauf ausbaust.
    return run_historical_cim_analysis(user_input=user_input, cim_root=CIM_ROOT)


@tool("powerfactory", return_direct=False)
def powerfactory_tool(user_input: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Platzhalter: PowerFactory Tool. Später hier echten Runner anbinden.
    """
    return {"status": "todo", "tool": "powerfactory", "input": user_input, "note": "PowerFactory noch nicht angebunden"}