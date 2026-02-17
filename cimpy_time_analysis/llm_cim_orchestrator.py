from llm_object_mapping import interpret_user_query
from cim_queries import query_equipment_power_over_time
from llm_result_agent import LLM_resultAgent
from cim_queries import summarize_powerflow


def handle_user_query(user_input, snapshot_cache, network_index):

    detected_types = interpret_user_query(user_input)

    if not detected_types:
        return "Ich konnte keinen Bezug zu Netzobjekten erkennen."

    if "SvPowerFlow" in detected_types and "PowerTransformer" in detected_types:

        results = query_equipment_power_over_time(
            snapshot_cache,
            network_index,
            equipment_type="PowerTransformer"
        )

    else:
        return f"Die erkannten Objekttypen {detected_types} werden aktuell noch nicht unterstützt."

    if not results:
        return "Es konnten keine passenden CIM-Snapshots ausgewertet werden."

    agent = LLM_resultAgent()
    summary = summarize_powerflow(results)
    return agent.summarize(summary, user_input)
