from cimpy.cimpy_time_analysis.llm_object_mapping import interpret_user_query
from cimpy.cimpy_time_analysis.cim_queries import (
    query_equipment_metric_over_time,
    query_equipment_voltage_over_time,
    summarize_metric,
    summarize_voltage
)
from cimpy.cimpy_time_analysis.llm_result_agent import LLM_resultAgent
from cimpy.cimpy_time_analysis.asset_resolver import resolve_equipment_from_query
from datetime import datetime

def _parse_dt_iso(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _filter_snapshot_cache_by_time(snapshot_cache: dict, start_iso: str | None, end_iso: str | None) -> dict:
    """
    Filtert snapshot_cache nach data["timestamp"] oder data["scenario_time"] (falls timestamp fehlt).
    Erwartet timezone-aware datetimes (ISO).
    """
    start_dt = _parse_dt_iso(start_iso) if start_iso else None
    end_dt = _parse_dt_iso(end_iso) if end_iso else None
    if not start_dt or not end_dt:
        return snapshot_cache

    out = {}
    for snap, data in snapshot_cache.items():
        ts = data.get("timestamp", None) or data.get("scenario_time", None)
        # ts kann datetime oder string sein
        if isinstance(ts, str):
            ts = _parse_dt_iso(ts)
        if ts is None:
            continue
        if start_dt <= ts < end_dt:
            out[snap] = data
    return out



def handle_user_query(user_input, snapshot_cache, network_index):
    parsed = interpret_user_query(user_input, network_index=network_index)

    equipment_detected = parsed.get("equipment_detected", [])
    state_detected = parsed.get("state_detected", [])
    metric = parsed.get("metric", None)
    equipment_selection = parsed.get("equipment_selection", [])
    time_start = parsed.get("time_start", None)
    time_end = parsed.get("time_end", None)
    time_label = parsed.get("time_label", None)

    # Snapshot-Cache auf Zeitraum reduzieren
    snapshot_cache = _filter_snapshot_cache_by_time(snapshot_cache, time_start, time_end)


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