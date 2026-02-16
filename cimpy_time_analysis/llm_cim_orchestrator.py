from llm_object_mapping import interpret_user_query
from cim_queries import query_total_powerflow_over_time
from cim_queries import query_trafo_power_over_time
from cim_queries import summarize_powerflow
from llm_result_agent import LLM_resultAgent


def handle_user_query(user_input, cim_snapshots):
    detected_types = interpret_user_query(user_input)

    if not detected_types:
        return "Ich konnte keinen Bezug zu Netzobjekten erkennen."

    # Aktuell: Trafo / Leistung → PowerFlow-Zeitreihe
    if "SvPowerFlow" in detected_types and "PowerTransformer" in detected_types:
        results = query_trafo_power_over_time(cim_snapshots)

    elif "SvPowerFlow" in detected_types:
        results = query_total_powerflow_over_time(cim_snapshots)

    else:
        return f"Die erkannten Objekttypen {detected_types} werden aktuell noch nicht unterstützt."

    if not results:
        return "Es konnten keine passenden CIM-Snapshots ausgewertet werden."

    agent = LLM_resultAgent()
    summary = summarize_powerflow(results)
    return agent.summarize(summary, user_input)

    
