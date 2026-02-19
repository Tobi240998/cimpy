from llm_object_mapping import interpret_user_query
from cim_queries import (
    query_equipment_power_over_time,
    query_equipment_voltage_over_time,
    summarize_powerflow,
    summarize_voltage
)
from llm_result_agent import LLM_resultAgent
from asset_resolver import resolve_equipment_from_query


def handle_user_query(user_input, snapshot_cache, network_index):
    detected_types = interpret_user_query(user_input) #Liste der Objekte / Berechnungen, die aus dem User Input abgeleitet werden

    if not detected_types:
        return "Ich konnte keinen Bezug zu Netzobjekten erkennen."

    agent = LLM_resultAgent()

    # ---------------------------------------------------------
    # 1) falls Trafo + Leistung (SvPowerFlow) in detected types
    # ---------------------------------------------------------
    if "SvPowerFlow" in detected_types and "PowerTransformer" in detected_types:
        #Sucht den richtigen Trafo 
        trafo_obj, debug = resolve_equipment_from_query(
            user_input=user_input,
            equipment_type="PowerTransformer",
            network_index=network_index
        )

        if not trafo_obj:
            return (
                "Ich konnte den gewünschten Trafo nicht eindeutig zuordnen. "
                "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20'). "
                f"(Matching-Methode: {debug.get('method')})"
            )

        results = query_equipment_power_over_time( #Sammelt die Scheinleistung zu jedem Zeitpunkt
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=trafo_obj
        )

        if not results:
            return "Keine SV-Leistungswerte für diesen Trafo in den Snapshots gefunden."

        summary = summarize_powerflow(results) #bereitet die Ergebnisse auf und gibt Minimum- / Maximumwerte und Durschnitt an 
        return agent.summarize(summary, user_input) #Zusammenfassung durch LLM-Agenten

    # ---------------------------------------------------------
    # 2) falls Trafo + Spannung (SvVoltage) in detected types - gleicher Aufbau der Funktion wie bei Trafo + Leistung
    # ---------------------------------------------------------
    if "SvVoltage" in detected_types and "PowerTransformer" in detected_types:

        #Sucht den richtigen Trafo
        trafo_obj, debug = resolve_equipment_from_query(
            user_input=user_input,
            equipment_type="PowerTransformer",
            network_index=network_index
        )

        if not trafo_obj:
            return (
                "Ich konnte den gewünschten Trafo nicht eindeutig zuordnen. "
                "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20'). "
                f"(Matching-Methode: {debug.get('method')})"
            )

        results = query_equipment_voltage_over_time(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=trafo_obj
        )

        if not results:
            return (
                "Keine SV-Spannungswerte für die Trafo-Knoten gefunden. "
                "Prüfe bitte, ob SvVoltage vorhanden ist und das Mapping "
                "Terminal→ConnectivityNode→TopologicalNode verfügbar ist."
            )

        summary = summarize_voltage(results)
        return agent.summarize(summary, user_input)

    # ---------------------------------------------------------
    # Default / nicht unterstützt
    # ---------------------------------------------------------
    return f"Die erkannten Objekttypen {detected_types} werden aktuell noch nicht unterstützt."
