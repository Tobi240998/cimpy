from cimpy.cimpy_time_analysis.llm_object_mapping import interpret_user_query
from cimpy.cimpy_time_analysis.cim_queries import (
    query_equipment_metric_over_time,
    query_equipment_voltage_over_time,
    summarize_metric,
    summarize_voltage
)
from cimpy.cimpy_time_analysis.llm_result_agent import LLM_resultAgent
from cimpy.cimpy_time_analysis.asset_resolver import resolve_equipment_from_query


def handle_user_query(user_input, snapshot_cache, network_index):
    parsed = interpret_user_query(user_input, network_index=network_index)

    equipment_detected = parsed.get("equipment_detected", [])
    state_detected = parsed.get("state_detected", [])
    metric = parsed.get("metric", None)
    equipment_selection = parsed.get("equipment_selection", [])

        # 1) Eine Selection erwarten (aktuell wählst du genau eins)
    if not equipment_selection:
        return (
                "Ich konnte das gewünschte Equipment nicht eindeutig zuordnen. "
                "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20' oder 'Load 27')."
        )

    sel = equipment_selection[0]
    equipment_type = sel["equipment_type"]
    equipment_key = sel["equipment_key"]

    # 2) Hier der wichtige Fix: echtes Objekt aus dem Index holen
    equipment_obj = network_index["equipment_name_index"][equipment_type][equipment_key]

    agent = LLM_resultAgent()

    # Jetzt hast du Equipment+State getrennt UND die konkrete Auswahl inkl. ID:
    # equipment_obj = [{"equipment_type","equipment_key","equipment_name","equipment_id"}, ...]

    # ... ab hier ganz normal weiterverarbeiten, kein early return nötig ...
    # z.B.:
    # for sel in equipment_obj:
    #     equipment_obj = network_index["equipment_name_index"][sel["equipment_type"]][sel["equipment_key"]]
    #     ...

    # Debug:
    print("equipment_detected:", equipment_detected)
    print("state_detected:", state_detected)
    print("metric:", metric)
    print("equipment_obj:", equipment_obj)

  

    # ---------------------------------------------------------
    # 1) Equipment-Typ ableiten (Trafo / Verbraucher)
    # ---------------------------------------------------------
    equipment_type = None
    if "PowerTransformer" in equipment_detected:
        equipment_type = "PowerTransformer"
    elif "ConformLoad" in equipment_detected:
        equipment_type = "ConformLoad"

   
    if not equipment_obj:
        return (
            "Ich konnte das gewünschte Equipment nicht eindeutig zuordnen. "
            "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20' oder 'Load 27'). "
        )

    # ---------------------------------------------------------
    # 2) falls Leistung (SvPowerFlow) in detected types
    # ---------------------------------------------------------
    if "SvPowerFlow" in state_detected:

        # -----------------------------------------------------
        # Unterschiedliche Default-Metrik:
        # Trafo -> S (MVA)
        # Load  -> P (MW)
        # -----------------------------------------------------
        if metric is None:
            if equipment_type == "PowerTransformer":
                metric = "S"
            elif equipment_type == "ConformLoad":
                metric = "P"
            else:
                metric = "S"

        metric = metric.upper()

        print("equipment_obj type:", type(equipment_obj))
        print("equipment_obj name:", getattr(equipment_obj, "name", None))
        print("equipment_obj id:", getattr(equipment_obj, "mRID", None), getattr(equipment_obj, "rdfId", None))




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
    if "SvVoltage" in state_detected:

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
    return f"Die erkannten Objekttypen {equipment_detected} werden aktuell noch nicht unterstützt."