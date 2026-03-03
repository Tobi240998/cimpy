from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.runner import run_historical_cim_analysis

def historic_executor(user_input: str):
    return run_historical_cim_analysis(user_input=user_input, cim_root=CIM_ROOT)

def pf_executor(user_input: str):
    return {"note": "PowerFactory (TODO)", "input": user_input}