import math

from cimpy.cimpy_time_analysis.cim_topology_graph import (
    get_equipment_neighbors,
    get_connected_component_for_equipment,
    find_shortest_path_between_equipments,
)

#Normalisierung der Terminal ID auf einheitliches Format 
def _canonical_id(value):
    if value is None:
        return None
    if not isinstance(value, str):
        value = getattr(value, "mRID", None)
        if value is None:
            return None
    s = value.strip()
    if "#" in s:
        s = s.split("#")[-1]
    if s.lower().startswith("urn:uuid:"):
        s = s.split(":", 2)[-1]
    s = s.strip()
    if s and not s.startswith("_"):
        s = "_" + s
    return s.lower()


def _get_rated_s_mva(equipment_obj):
    """
    Holt ratedS (MVA) aus EQ-Profil, falls vorhanden.
    Für PowerTransformer: ratedS liegt typischerweise an PowerTransformerEnd.

    Für EnergyConsumer etc. gibt es meist keine ratedS -> None.
    """
    if equipment_obj is None:
        return None

    ends = getattr(equipment_obj, "PowerTransformerEnd", None)
    if not ends:
        return None

    rated_vals = []
    for end in ends:
        rated = getattr(end, "ratedS", None)
        if rated is None:
            continue
        try:
            rated_vals.append(float(rated))
        except Exception:
            continue

    rated_vals = [v for v in rated_vals if v > 0]
    if not rated_vals:
        return None

    # robust: Maximum über alle Enden (falls unterschiedlich / nur auf einer Seite gepflegt)
    return max(rated_vals)


def query_equipment_metric_over_time(
    snapshot_cache,
    network_index,
    equipment_obj,
    metric: str = "S"
    ):
   
    if equipment_obj is None:
        return []

    equipment_type = equipment_obj.__class__.__name__
    equipment_name = getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN"))  #gibt Equipment-Namen aus, falls dieser nicht gefunden wird -> mRID

    metric = (metric or "S").upper().strip()
    if metric not in {"S", "P", "Q"}:
        metric = "S"

    # ratedS nur relevant für Auslastung in % bei S und PowerTransformer
    rated_mva = None
    if metric == "S" and equipment_type == "PowerTransformer":
        rated_mva = _get_rated_s_mva(equipment_obj)

    results = []

    for snapshot, data in snapshot_cache.items():
        flows = data.get("flows", [])  #zieht sich die SvPowerFlow Objekte aus den Snapshots
        if not flows:
            continue

        ts = data.get("timestamp", None)  #zieht sich die Timestamps aus den Snapshots
        ts_str = data.get("timestamp_str", None)

        # Wir merken uns pro Snapshot den "besten" Terminal-Wert (größter Betrag, wird hier "zurückgesetzt")
        best_val = None
        best_abs = -1.0

        for flow in flows:
            terminal = getattr(flow, "Terminal", None)  #Zuordnung des Terminals zum SvPowerFlow
            terminal_id = _canonical_id(terminal)  #Normalisierung auf einheitliches Format
            if not terminal_id:
                continue

            eq = network_index["terminals_to_equipment"].get(terminal_id)  #sucht das Equipment zum jeweiligen Terminal raus
            if eq != equipment_obj:  #Prüfung, ob gefundenes Equipment zum gesuchten passt -> falls ja, werden Werte berechnet, sonst skippen
                continue

            p = getattr(flow, "p", 0.0)
            q = getattr(flow, "q", 0.0)

            #Berechnung bzw. Ausgabe der richtigen Einheit
            if metric == "S":
                val = math.sqrt(p**2 + q**2)
            elif metric == "P":
                val = float(p)
            else:  # metric == "Q"
                val = float(q)

            #Prüfung, ob größter Wert und falls ja, Speicherung
            a = abs(val)
            if a > best_abs:
                best_abs = a
                best_val = val

        if best_val is None:
            continue

        row = {
            "snapshot": snapshot,
            "timestamp": ts,
            "timestamp_str": ts_str,
            "equipment": equipment_name,
            "type": equipment_type,
            "metric": metric
        }

        # je nach Metrik den passenden Key setzen
        if metric == "S":
            row["apparent_power_MVA"] = float(best_val)
            row["rated_MVA"] = rated_mva  #Nennleistung (MVA)
            row["loading_percent"] = (float(best_val) / rated_mva * 100.0) if (rated_mva is not None and rated_mva > 0) else None  #Auslastung (%)
        elif metric == "P":
            row["active_power_MW"] = float(best_val)
        else:
            row["reactive_power_MVAr"] = float(best_val)

        results.append(row)

    results.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"]))  #zeitliche Sortierung der Ergebnisse
    return results




def query_equipment_voltage_over_time(snapshot_cache, network_index, equipment_obj):  #bestimmt Spannung des gewählten Equipments; gleicher Aufbau wie bei query equipment metric over time, außer anderem Mapping und Ziehen der Spannung
    """
    FIX: Terminals werden NICHT aus equipment_obj.Terminals gezogen,
    sondern über network_index["equipment_to_terminal_ids"].

    Pfad:
      equipment_id -> terminal_ids -> connectivityNode_id -> topologicalNode_id -> SvVoltage.v

    Hinweis:
    - Diese Query ist typunabhängig: funktioniert für PowerTransformer UND EnergyConsumer -> kann noch erweitert werden,
      solange das Equipment Terminals hat und SvVoltage auf dem zugehörigen TopologicalNode existiert.
    """

    if equipment_obj is None:
        return []

    equipment_type = equipment_obj.__class__.__name__
    equipment_name = getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN"))
    equipment_id = _canonical_id(getattr(equipment_obj, "mRID", None))

    terminal_ids = network_index.get("equipment_to_terminal_ids", {}).get(equipment_id, [])
    if not terminal_ids:
        return []

    results = []

    for snapshot, data in snapshot_cache.items():
        ts = data.get("timestamp", None)
        ts_str = data.get("timestamp_str", None)
        voltage_by_node = data.get("voltage_by_node", {})

        for terminal_id in terminal_ids:
            cn_id = network_index.get("terminal_to_connectivitynode", {}).get(terminal_id)
            if not cn_id:
                continue

            tn_id = network_index.get("connectivitynode_to_topologicalnode", {}).get(cn_id)
            if not tn_id:
                continue

            sv = voltage_by_node.get(tn_id)
            if not sv:
                continue

            v = getattr(sv, "v", None)
            if v is None:
                continue

            results.append({
                "snapshot": snapshot,
                "timestamp": ts,
                "timestamp_str": ts_str,
                "equipment": equipment_name,
                "type": equipment_type,
                "terminal_id": terminal_id,
                "connectivity_node_id": cn_id,
                "topological_node_id": tn_id,
                "voltage_kV": float(v)
            })

    results.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"], r["terminal_id"]))
    return results


def summarize_metric(results):
    if not results:
        return {"type": None, "message": "Keine Daten verfügbar"}

    metric = results[0]["metric"]  # P / Q / S

    if metric == "P":
        unit = "MW"
        key = "active_power_MW"
    elif metric == "Q":
        unit = "MVAr"
        key = "reactive_power_MVAr"
    else:
        unit = "MVA"
        key = "apparent_power_MVA"

    values = [r[key] for r in results if key in r]

    peak_entry = max(results, key=lambda r: r.get(key, float("-inf")))
    min_entry = min(results, key=lambda r: r.get(key, float("inf")))

    summary = {
        "metric": metric,
        "unit": unit,
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": sum(values) / len(values),
        "peak_value": peak_entry.get(key),
        "peak_timestamp": peak_entry.get("timestamp_str"),
        "min_value_at_time": min_entry.get(key),
        "min_timestamp": min_entry.get("timestamp_str"),
        "num_datapoints": len(values)
    }

    #Auslastung (%) nur bei S und wenn vorhanden
    if metric == "S":
        loading_values = [r["loading_percent"] for r in results if r.get("loading_percent") is not None]
        if loading_values:
            peak_loading_entry = max(
                [r for r in results if r.get("loading_percent") is not None],
                key=lambda r: r["loading_percent"]
            )
            min_loading_entry = min(
                [r for r in results if r.get("loading_percent") is not None],
                key=lambda r: r["loading_percent"]
            )

            summary.update({
                "rated_MVA": results[0].get("rated_MVA", None),  #Nennleistung, wird für Auslastung genutzt
                "loading_unit": "%",
                "loading_min_percent": min(loading_values),
                "loading_max_percent": max(loading_values),
                "loading_mean_percent": sum(loading_values) / len(loading_values),
                "loading_peak_percent": peak_loading_entry["loading_percent"],
                "loading_peak_snapshot": peak_loading_entry.get("snapshot"),
                "loading_peak_timestamp": peak_loading_entry.get("timestamp_str"),
                "loading_min_percent_at_time": min_loading_entry["loading_percent"],
                "loading_min_snapshot": min_loading_entry.get("snapshot"),
                "loading_min_timestamp": min_loading_entry.get("timestamp_str"),
            })

    return summary




def summarize_voltage(results):  #gleiche Vorgehensweise wie bei summarize_metric
    if not results:
        return {"type": None, "message": "Keine Daten verfügbar"}

    values = [r["voltage_kV"] for r in results]
    max_entry = max(results, key=lambda r: r["voltage_kV"])
    min_entry = min(results, key=lambda r: r["voltage_kV"])

    return {
        "type": results[0]["type"],
        "unit": "kV",
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": sum(values) / len(values),
        "max_value_at_time": max_entry["voltage_kV"],
        "max_snapshot": max_entry["snapshot"],
        "max_timestamp": max_entry.get("timestamp_str"),
        "max_equipment": max_entry["equipment"],
        "max_terminal_id": max_entry.get("terminal_id"),
        "max_topological_node_id": max_entry.get("topological_node_id"),
        "min_value_at_time": min_entry["voltage_kV"],
        "min_snapshot": min_entry["snapshot"],
        "min_timestamp": min_entry.get("timestamp_str"),
        "min_equipment": min_entry["equipment"],
        "min_terminal_id": min_entry.get("terminal_id"),
        "min_topological_node_id": min_entry.get("topological_node_id"),
        "num_datapoints": len(results)
    }


# =============================================================================
# TOPOLOGIE-QUERIES (neu)
# =============================================================================

def _get_topology_graph_from_index(network_index, level: str = "connectivity"):
    if level == "connectivity":
        return (
            network_index.get("topology_graph_connectivity")
            or network_index.get("topology_graph")
        )
    if level == "topological":
        return network_index.get("topology_graph_topological")
    return None


def query_equipment_topology_neighbors(
    network_index,
    equipment_obj,
    level: str = "connectivity",
    allowed_neighbor_classes=None,
):
    """
    Liefert direkte topologische Nachbarn eines Equipments.

    Parameters
    ----------
    network_index : dict
        Globaler Netzindex.
    equipment_obj : CIM-Objekt
        Gewähltes Equipment.
    level : str
        "connectivity" oder "topological"
    allowed_neighbor_classes : list[str] | None
        Optionaler Klassenfilter für Nachbarn.
    """
    if equipment_obj is None:
        return []

    G = _get_topology_graph_from_index(network_index, level=level)
    if G is None:
        return []

    equipment_name = getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN"))
    equipment_type = equipment_obj.__class__.__name__
    equipment_id = _canonical_id(getattr(equipment_obj, "mRID", None))

    neighbors = get_equipment_neighbors(
        G=G,
        equipment_obj_or_id=equipment_obj,
        allowed_neighbor_classes=allowed_neighbor_classes,
    )

    results = []
    for nbr in neighbors:
        results.append({
            "equipment": equipment_name,
            "type": equipment_type,
            "equipment_id": equipment_id,
            "neighbor_equipment": nbr.get("name"),
            "neighbor_type": nbr.get("cim_class"),
            "neighbor_equipment_id": nbr.get("equipment_id"),
            "neighbor_degree": nbr.get("degree"),
            "edge_count": nbr.get("edge_count"),
            "edge_types": nbr.get("edge_types"),
            "shared_topology_node_ids": nbr.get("shared_topology_node_ids", []),
            "graph_level": level,
        })

    results.sort(key=lambda r: ((r["neighbor_equipment"] or ""), r["neighbor_equipment_id"] or ""))
    return results


def query_equipment_connected_component(
    network_index,
    equipment_obj,
    level: str = "connectivity",
):
    """
    Liefert die zusammenhängende Komponente des Equipments im Topologiegraphen.
    """
    if equipment_obj is None:
        return {
            "equipment": None,
            "type": None,
            "equipment_id": None,
            "component_size": 0,
            "node_kind_counts": {},
            "equipment_nodes": [],
            "graph_level": level,
        }

    G = _get_topology_graph_from_index(network_index, level=level)
    if G is None:
        return {
            "equipment": getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN")),
            "type": equipment_obj.__class__.__name__,
            "equipment_id": _canonical_id(getattr(equipment_obj, "mRID", None)),
            "component_size": 0,
            "node_kind_counts": {},
            "equipment_nodes": [],
            "graph_level": level,
        }

    component = get_connected_component_for_equipment(G=G, equipment_obj_or_id=equipment_obj)

    return {
        "equipment": getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN")),
        "type": equipment_obj.__class__.__name__,
        "equipment_id": _canonical_id(getattr(equipment_obj, "mRID", None)),
        "component_size": component.get("component_size", 0),
        "node_kind_counts": component.get("node_kind_counts", {}),
        "equipment_nodes": component.get("equipment_nodes", []),
        "graph_level": level,
    }


def query_shortest_topology_path(
    network_index,
    source_equipment_obj,
    target_equipment_obj,
    level: str = "connectivity",
):
    """
    Kürzester topologischer Pfad zwischen zwei Equipments.
    Ist schon vorbereitet, auch wenn du ihn im Orchestrator zunächst noch nicht nutzt.
    """
    if source_equipment_obj is None or target_equipment_obj is None:
        return {
            "found": False,
            "reason": "missing_equipment",
            "graph_level": level,
            "source_equipment": None,
            "target_equipment": None,
            "path": [],
            "path_details": [],
        }

    G = _get_topology_graph_from_index(network_index, level=level)
    if G is None:
        return {
            "found": False,
            "reason": "graph_missing",
            "graph_level": level,
            "source_equipment": getattr(source_equipment_obj, "name", getattr(source_equipment_obj, "mRID", "UNKNOWN")),
            "target_equipment": getattr(target_equipment_obj, "name", getattr(target_equipment_obj, "mRID", "UNKNOWN")),
            "path": [],
            "path_details": [],
        }

    result = find_shortest_path_between_equipments(
        G=G,
        source_equipment_obj_or_id=source_equipment_obj,
        target_equipment_obj_or_id=target_equipment_obj,
    )

    result["graph_level"] = level
    result["source_equipment"] = getattr(source_equipment_obj, "name", getattr(source_equipment_obj, "mRID", "UNKNOWN"))
    result["target_equipment"] = getattr(target_equipment_obj, "name", getattr(target_equipment_obj, "mRID", "UNKNOWN"))
    result["source_type"] = source_equipment_obj.__class__.__name__
    result["target_type"] = target_equipment_obj.__class__.__name__

    return result


# =============================================================================
# TOPOLOGIE-SUMMARIES (neu)
# =============================================================================

def summarize_topology_neighbors(results):
    if not results:
        return {
            "type": "topology_neighbors",
            "message": "Keine topologischen Nachbarn gefunden",
            "num_neighbors": 0,
            "graph_level": None,
        }

    neighbor_names = [r["neighbor_equipment"] for r in results if r.get("neighbor_equipment")]
    neighbor_types = [r["neighbor_type"] for r in results if r.get("neighbor_type")]
    degrees = [r["neighbor_degree"] for r in results if r.get("neighbor_degree") is not None]

    type_counts = {}
    for t in neighbor_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    unique_shared_nodes = sorted({
        topo_id
        for r in results
        for topo_id in r.get("shared_topology_node_ids", [])
        if topo_id
    })

    return {
        "type": "topology_neighbors",
        "equipment": results[0].get("equipment"),
        "equipment_type": results[0].get("type"),
        "equipment_id": results[0].get("equipment_id"),
        "graph_level": results[0].get("graph_level"),
        "num_neighbors": len(results),
        "neighbor_names": neighbor_names,
        "neighbor_type_counts": type_counts,
        "max_neighbor_degree": max(degrees) if degrees else None,
        "min_neighbor_degree": min(degrees) if degrees else None,
        "shared_topology_node_ids": unique_shared_nodes,
        "neighbors": results,
    }


def summarize_topology_component(component_result, max_list_entries: int = 25):
    if not component_result or not component_result.get("equipment"):
        return {
            "type": "topology_component",
            "message": "Keine zusammenhängende Komponente gefunden",
            "component_size": 0,
            "equipment_count": 0,
        }

    equipment_nodes = component_result.get("equipment_nodes", [])
    equipment_count = len(equipment_nodes)

    listed_equipment = equipment_nodes[:max_list_entries]
    listed_names = [e.get("name") for e in listed_equipment if e.get("name")]

    equipment_type_counts = {}
    for e in equipment_nodes:
        cls = e.get("cim_class")
        if not cls:
            continue
        equipment_type_counts[cls] = equipment_type_counts.get(cls, 0) + 1

    return {
        "type": "topology_component",
        "equipment": component_result.get("equipment"),
        "equipment_type": component_result.get("type"),
        "equipment_id": component_result.get("equipment_id"),
        "graph_level": component_result.get("graph_level"),
        "component_size": component_result.get("component_size", 0),
        "equipment_count": equipment_count,
        "node_kind_counts": component_result.get("node_kind_counts", {}),
        "equipment_type_counts": equipment_type_counts,
        "listed_equipment_names": listed_names,
        "listed_equipment": listed_equipment,
        "truncated": equipment_count > max_list_entries,
    }


def summarize_topology_path(path_result):
    if not path_result:
        return {
            "type": "topology_path",
            "message": "Kein Pfad-Ergebnis vorhanden",
            "found": False,
        }

    if not path_result.get("found"):
        return {
            "type": "topology_path",
            "found": False,
            "reason": path_result.get("reason"),
            "graph_level": path_result.get("graph_level"),
            "source_equipment": path_result.get("source_equipment"),
            "target_equipment": path_result.get("target_equipment"),
            "path_length": 0,
            "path_labels": [],
        }

    path_details = path_result.get("path_details", [])
    path_labels = []

    for item in path_details:
        name = item.get("name")
        cim_class = item.get("cim_class")
        kind = item.get("kind")

        if name and cim_class:
            path_labels.append(f"{name} ({cim_class})")
        elif name:
            path_labels.append(name)
        elif cim_class:
            path_labels.append(f"{item.get('node_id')} ({cim_class}/{kind})")
        else:
            path_labels.append(item.get("node_id"))

    return {
        "type": "topology_path",
        "found": True,
        "reason": None,
        "graph_level": path_result.get("graph_level"),
        "source_equipment": path_result.get("source_equipment"),
        "target_equipment": path_result.get("target_equipment"),
        "source_type": path_result.get("source_type"),
        "target_type": path_result.get("target_type"),
        "path_length": path_result.get("path_length", 0),
        "path_labels": path_labels,
        "path_details": path_details,
    }