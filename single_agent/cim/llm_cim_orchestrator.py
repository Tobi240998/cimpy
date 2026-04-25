from datetime import datetime

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
    get_component_equipment_objects,
    get_neighbor_equipment_objects,
    aggregate_metric_over_equipment_set,
)
from cimpy.cimpy_time_analysis.llm_result_agent import LLM_resultAgent


def _parse_dt_iso(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _filter_snapshot_cache_by_time(snapshot_cache: dict, start_iso: str | None, end_iso: str | None) -> dict:
    start_dt = _parse_dt_iso(start_iso) if start_iso else None
    end_dt = _parse_dt_iso(end_iso) if end_iso else None
    if not start_dt or not end_dt:
        return snapshot_cache

    out = {}
    for snap, data in snapshot_cache.items():
        ts = data.get("timestamp", None) or data.get("scenario_time", None)
        if isinstance(ts, str):
            ts = _parse_dt_iso(ts)
        if ts is None:
            continue
        if start_dt <= ts < end_dt:
            out[snap] = data
    return out

# Refactoring: Heuristik
def _detect_topology_intent(user_input: str, analysis_plan: dict | None = None) -> dict:
    if analysis_plan:
        topology_scope = analysis_plan.get("topology_scope", "none")
        needs_topology_graph = bool(analysis_plan.get("needs_topology_graph", False))
        graph_level = analysis_plan.get("graph_level", "connectivity") or "connectivity"

        if needs_topology_graph and topology_scope in {"neighbors", "component"}:
            return {
                "is_topology": True,
                "intent": topology_scope,
                "graph_level": graph_level,
            }

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


def _resolve_equipment_from_selection(parsed: dict, network_index: dict):
    equipment_selection = parsed.get("equipment_selection", []) or []
    if not equipment_selection:
        return None, None, None

    sel = equipment_selection[0]
    equipment_type = sel["equipment_type"]
    equipment_key = sel["equipment_key"]

    equipment_obj = network_index["equipment_name_index"][equipment_type][equipment_key]
    return equipment_obj, equipment_type, equipment_key

# Refactoring: nochmal prüfen, ob Einheiten wirklich vorher festgesetzt werden müssen 
def _default_metric_for_equipment_type(equipment_type: str | None):
    if equipment_type == "PowerTransformer":
        return "S"
    if equipment_type == "ConformLoad":
        return "P"
    return "S"


def _ensure_parsed_query(
    user_input: str,
    network_index: dict,
    parsed_query: dict | None = None,
):
    if parsed_query is not None:
        return parsed_query
    return interpret_user_query(user_input, network_index=network_index)


def handle_user_query(
    user_input,
    snapshot_cache,
    network_index,
    parsed_query=None,
    analysis_plan=None,
):
    parsed = _ensure_parsed_query(
        user_input=user_input,
        network_index=network_index,
        parsed_query=parsed_query,
    )

    equipment_detected = parsed.get("equipment_detected", [])
    state_detected = parsed.get("state_detected", [])
    metric = parsed.get("metric", None)
    equipment_selection = parsed.get("equipment_selection", [])
    time_start = parsed.get("time_start", None)
    time_end = parsed.get("time_end", None)
    time_label = parsed.get("time_label", None)

    snapshot_cache = _filter_snapshot_cache_by_time(snapshot_cache, time_start, time_end)

    if not equipment_selection:
        return (
            "Ich konnte das gewünschte Equipment nicht eindeutig zuordnen. "
            "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20' oder 'Load 27')."
        )

    equipment_obj, equipment_type, equipment_key = _resolve_equipment_from_selection(
        parsed=parsed,
        network_index=network_index,
    )

    agent = LLM_resultAgent()

    print("analysis_plan:", analysis_plan)
    print("equipment_detected:", equipment_detected)
    print("state_detected:", state_detected)
    print("metric:", metric)
    print("equipment_obj:", equipment_obj)
    print("time_label:", time_label)

    if not equipment_obj:
        return (
            "Ich konnte das gewünschte Equipment nicht eindeutig zuordnen. "
            "Bitte prüfe die Schreibweise (z.B. 'Trafo 19 - 20' oder 'Load 27')."
        )

    topology_intent = _detect_topology_intent(
        user_input=user_input,
        analysis_plan=analysis_plan,
    )
    print("topology_intent:", topology_intent)

    # ---------------------------------------------------------
    # Kombination: Topologie -> Equipment-Menge -> bestehende State-Query
    # ---------------------------------------------------------
    if (
        analysis_plan
        and analysis_plan.get("query_mode") == "topology_plus_state"
        and "SvPowerFlow" in state_detected
        and topology_intent.get("is_topology")
    ):
        graph_level = topology_intent.get("graph_level", "topological")
        topology_scope = topology_intent.get("intent")
        target_types = analysis_plan.get("target_equipment_types") or []
        target_equipment_type = target_types[0] if target_types else None

        if metric is None:
            metric = analysis_plan.get("metric_hint") or "P"

        aggregation = analysis_plan.get("aggregation") or "max"

        if topology_scope == "component":
            equipment_set = get_component_equipment_objects(
                network_index=network_index,
                reference_equipment_obj=equipment_obj,
                level=graph_level,
                target_equipment_type=target_equipment_type,
            )
        elif topology_scope == "neighbors":
            equipment_set = get_neighbor_equipment_objects(
                network_index=network_index,
                reference_equipment_obj=equipment_obj,
                level=graph_level,
                target_equipment_type=target_equipment_type,
            )
        else:
            equipment_set = []

        result = aggregate_metric_over_equipment_set(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_objects=equipment_set,
            metric=metric,
            aggregation=aggregation,
        )

        return agent.summarize(result, user_input)

    # ---------------------------------------------------------
    # Reine Topologie
    # ---------------------------------------------------------
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
    # Leistung / Auslastung
    # ---------------------------------------------------------
    if "SvPowerFlow" in state_detected:
        if metric is None:
            metric = _default_metric_for_equipment_type(equipment_type)

        metric = metric.upper()

        print("equipment_obj type:", type(equipment_obj))
        print("equipment_obj name:", getattr(equipment_obj, "name", None))
        print("equipment_obj id:", getattr(equipment_obj, "mRID", None), getattr(equipment_obj, "rdfId", None))

        results = query_equipment_metric_over_time(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=equipment_obj,
            metric=metric,
        )

        if not results:
            return "Keine SV-Leistungswerte für dieses Equipment in den Snapshots gefunden."

        summary = summarize_metric(results)
        return agent.summarize(summary, user_input)

    # ---------------------------------------------------------
    # Spannung
    # ---------------------------------------------------------
    if "SvVoltage" in state_detected:
        results = query_equipment_voltage_over_time(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=equipment_obj,
        )

        if not results:
            return (
                "Keine SV-Spannungswerte für die Equipment-Knoten gefunden. "
                "Prüfe bitte, ob SvVoltage vorhanden ist und das Mapping "
                "Terminal→ConnectivityNode→TopologicalNode verfügbar ist."
            )

        summary = summarize_voltage(results)
        return agent.summarize(summary, user_input)

    return f"Die erkannten Objekttypen {equipment_detected} werden aktuell noch nicht unterstützt."