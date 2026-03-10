from cimpy.cimpy_time_analysis.llm_object_mapping import interpret_user_query
from cimpy.cimpy_time_analysis.cim_queries import (
    query_equipment_metric_over_time,
    query_equipment_voltage_over_time,
    summarize_metric,
    summarize_voltage,
    query_equipment_topology_neighbors,
    query_equipment_connected_component,
    summarize_topology_neighbors,
    summarize_topology_component,
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


def _detect_topology_intent(user_input: str) -> dict:
    """
    Sehr einfache, rein additive Heuristik für Topologiefragen.
    Verändert die bestehende LLM-Logik nicht, sondern ergänzt nur
    einen zusätzlichen Pfad für klar topologische Fragen.

    Returns
    -------
    dict:
        {
            "is_topology": bool,
            "intent": "neighbors" | "component" | None,
            "graph_level": "connectivity" | "topological",
        }
    """
    text = (user_input or "").lower()

    graph_level = "connectivity"
    if "topological node" in text or "topologicalnode" in text or "topolog" in text:
        graph_level = "topological"

    neighbor_keywords = [
        "nachbar",
        "nachbarn",
        "benachbart",
        "direkt verbunden",
        "direkt angrenz",
        "angeschlossen an",
        "hängt an",
        "womit verbunden",
        "welche objekte sind mit",
        "was ist mit",
        "verbundene objekte",
    ]

    component_keywords = [
        "komponente",
        "zusammenhäng",
        "connected component",
        "insel",
        "teilnetz",
        "netzbereich",
        "netzsegment",
        "alles was verbunden ist",
        "alle verbundenen",
    ]

    for kw in component_keywords:
        if kw in text:
            return {
                "is_topology": True,
                "intent": "component",
                "graph_level": graph_level,
            }

    for kw in neighbor_keywords:
        if kw in text:
            return {
                "is_topology": True,
                "intent": "neighbors",
                "graph_level": graph_level,
            }

    return {
        "is_topology": False,
        "intent": None,
        "graph_level": graph_level,
    }


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

    # Debug:
    print("equipment_detected:", equipment_detected)
    print("state_detected:", state_detected)
    print("metric:", metric)
    print("equipment_obj:", equipment_obj)

    if not equipment_obj:
        return (
            "Ich konnte das gewünschte Equipment nicht eindeutig zuordnen. "
            "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20' oder 'Load 27'). "
        )

    # ---------------------------------------------------------
    # Topologie-Intent (neu, additive Ergänzung)
    # ---------------------------------------------------------
    topology_intent = _detect_topology_intent(user_input)
    print("topology_intent:", topology_intent)

    if topology_intent.get("is_topology"):
        graph_level = topology_intent.get("graph_level", "connectivity")
        intent = topology_intent.get("intent")

        if intent == "neighbors":
            results = query_equipment_topology_neighbors(
                network_index=network_index,
                equipment_obj=equipment_obj,
                level=graph_level,
                allowed_neighbor_classes=None,
            )

            if not results:
                return (
                    f"Ich habe für {getattr(equipment_obj, 'name', getattr(equipment_obj, 'mRID', 'UNKNOWN'))} "
                    f"keine direkten topologischen Nachbarn im {graph_level}-Graphen gefunden."
                )

            summary = summarize_topology_neighbors(results)
            return agent.summarize(summary, user_input)

        if intent == "component":
            result = query_equipment_connected_component(
                network_index=network_index,
                equipment_obj=equipment_obj,
                level=graph_level,
            )

            if not result or result.get("component_size", 0) == 0:
                return (
                    f"Ich habe für {getattr(equipment_obj, 'name', getattr(equipment_obj, 'mRID', 'UNKNOWN'))} "
                    f"keine zusammenhängende Komponente im {graph_level}-Graphen gefunden."
                )

            summary = summarize_topology_component(result)
            return agent.summarize(summary, user_input)

    # ---------------------------------------------------------
    # falls Leistung (SvPowerFlow) in detected types
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

        results = query_equipment_metric_over_time(  # Sammelt die Metrik-Werte zu jedem Zeitpunkt
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=equipment_obj,
            metric=metric
        )

        if not results:
            return "Keine SV-Leistungswerte für dieses Equipment in den Snapshots gefunden."

        summary = summarize_metric(results)  # bereitet die Ergebnisse auf und gibt Minimum- / Maximumwerte und Durschnitt an
        return agent.summarize(summary, user_input)  # Zusammenfassung durch LLM-Agenten

    # ---------------------------------------------------------
    # falls Spannung (SvVoltage) in detected types
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