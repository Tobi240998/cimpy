from llm_object_mapping import interpret_user_query
from cim_queries import query_equipment_power_over_time, summarize_powerflow
from llm_result_agent import LLM_resultAgent
from asset_resolver import resolve_equipment_from_query


def handle_user_query(user_input, snapshot_cache, network_index):
    detected_types = interpret_user_query(user_input)

    if not detected_types:
        return "Ich konnte keinen Bezug zu Netzobjekten erkennen."

    # Aktuell: Trafo + Leistung
    if "SvPowerFlow" in detected_types and "PowerTransformer" in detected_types:

        trafo_obj, debug = resolve_equipment_from_query(
            user_input=user_input,
            equipment_type="PowerTransformer",
            network_index=network_index
        )

        if not trafo_obj:
            # Debug-Info bewusst knapp, aber hilfreich:
            return (
                "Ich konnte den gewünschten Trafo nicht eindeutig zuordnen. "
                "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20'). "
                f"(Matching-Methode: {debug.get('method')})"
            )

        results = query_equipment_power_over_time(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=trafo_obj
        )

    else:
        return f"Die erkannten Objekttypen {detected_types} werden aktuell noch nicht unterstützt."

    if not results:
        return "Es konnten keine passenden CIM-Snapshots ausgewertet werden (keine SV-Werte gefunden)."

    agent = LLM_resultAgent()
    summary = summarize_powerflow(results)
    return agent.summarize(summary, user_input)
