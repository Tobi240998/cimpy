from llm_object_mapping import interpret_user_query
from cim_queries import (
    query_equipment_metric_over_time,
    query_equipment_voltage_over_time,
    summarize_metric,
    summarize_voltage
)
from llm_result_agent import LLM_resultAgent
from asset_resolver import resolve_equipment_from_query


def handle_user_query(user_input, snapshot_cache, network_index):
    parsed = interpret_user_query(user_input) #Liste der Objekte / Berechnungen, die aus dem User Input abgeleitet werden
    detected_types = parsed.get("detected_types", [])
    metric = parsed.get("metric", None)

    if not detected_types:
        return "Ich konnte keinen Bezug zu Netzobjekten erkennen."

    agent = LLM_resultAgent()

    # ---------------------------------------------------------
    # 0) Default: wenn ein Equipment erkannt wurde, aber keine StateVariable
    #    -> wir gehen von Leistungsabfrage (SvPowerFlow) aus
    # ---------------------------------------------------------
    equipment_detected = any(t in detected_types for t in ["PowerTransformer", "ConformLoad"])
    state_detected = any(t in detected_types for t in ["SvPowerFlow", "SvVoltage"])

    if equipment_detected and not state_detected:
        detected_types.append("SvPowerFlow")  #Default auf Leistung, falls User nur "Load 27 ..." schreibt
        if metric is None:
            metric = "S"  #Default-Metrik: Scheinleistung

    # ---------------------------------------------------------
    # 1) Equipment-Typ ableiten (Trafo / Verbraucher)
    #    Falls kein Typ erkannt wurde, lassen wir den Resolver typfrei suchen.
    # ---------------------------------------------------------
    equipment_type = None
    if "PowerTransformer" in detected_types:
        equipment_type = "PowerTransformer"
    elif "ConformLoad" in detected_types:
        equipment_type = "ConformLoad"

    #Sucht das richtige Equipment (Trafo oder Verbraucher)
    equipment_obj, debug = resolve_equipment_from_query(
        user_input=user_input,
        equipment_type=equipment_type,
        network_index=network_index
    )

    if not equipment_obj:
        return (
            "Ich konnte das gewünschte Equipment nicht eindeutig zuordnen. "
            "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20' oder 'Load 27'). "
            f"(Matching-Methode: {debug.get('method')})"
        )

    # ---------------------------------------------------------
    # 2) falls Leistung (SvPowerFlow) in detected types
    #    -> Metrik wird aus User Input abgeleitet (S/P/Q); default ist S
    # ---------------------------------------------------------
    if "SvPowerFlow" in detected_types:

        metric = (metric or "S").upper()

        results = query_equipment_metric_over_time( #Sammelt die Metrik-Werte zu jedem Zeitpunkt
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=equipment_obj,
            metric=metric
        )

        if not results:
            return "Keine SV-Leistungswerte für dieses Equipment in den Snapshots gefunden."

        summary = summarize_metric(results) #bereitet die Ergebnisse auf und gibt Minimum- / Maximumwerte und Durschnitt an
        return agent.summarize(summary, user_input) #Zusammenfassung durch LLM-Agenten

    # ---------------------------------------------------------
    # 3) falls Spannung (SvVoltage) in detected types
    # ---------------------------------------------------------
    if "SvVoltage" in detected_types:

        results = query_equipment_voltage_over_time(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=equipment_obj
        )

        if not results:
            return (
                "Keine SV-Spannungswerte für die Equipment-Knoten gefunden. "
                "Prüfe bitte, ob SvVoltage vorhanden ist und das Mapping "
                "Terminal→ConnectivityNode→TopologicalNode verfügbar ist."
            )

        summary = summarize_voltage(results)
        return agent.summarize(summary, user_input)

    # ---------------------------------------------------------
    # Default / nicht unterstützt
    # ---------------------------------------------------------
    return f"Die erkannten Objekttypen {detected_types} werden aktuell noch nicht unterstützt."