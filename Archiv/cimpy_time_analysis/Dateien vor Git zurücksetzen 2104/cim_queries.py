import math

from cimpy.cimpy_time_analysis.cim_topology_graph import (
    get_equipment_neighbors,
    get_connected_component_for_equipment,
    find_shortest_path_between_equipments,
)


# =============================================================================
# Basis-Helper
# =============================================================================

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

    return max(rated_vals)


def _get_topology_graph_from_index(network_index, level: str = "connectivity"):
    if level == "connectivity":
        return (
            network_index.get("topology_graph_connectivity")
            or network_index.get("topology_graph")
        )
    if level == "topological":
        return network_index.get("topology_graph_topological")
    return None


def _build_equipment_object_lookup(network_index):
    lookup = {}
    equipment_name_index = (network_index or {}).get("equipment_name_index", {}) or {}
    for _, type_space in equipment_name_index.items():
        for _, obj in type_space.items():
            eq_id = _canonical_id(getattr(obj, "mRID", None))
            if eq_id:
                lookup[eq_id] = obj
    return lookup



# =============================================================================
# Einzel-Equipment: State Queries
# =============================================================================

def query_equipment_metric_over_time(
    snapshot_cache,
    network_index,
    equipment_obj,
    metric: str = "S"
):
    if equipment_obj is None:
        return []

    equipment_type = equipment_obj.__class__.__name__
    equipment_name = getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN"))

    metric = (metric or "S").upper().strip()
    if metric not in {"S", "P", "Q"}:
        metric = "S"

    rated_mva = None
    if metric == "S" and equipment_type == "PowerTransformer":
        rated_mva = _get_rated_s_mva(equipment_obj)

    results = []

    for snapshot, data in snapshot_cache.items():
        flows = data.get("flows", [])
        if not flows:
            continue

        ts = data.get("timestamp", None)
        ts_str = data.get("timestamp_str", None)

        best_val = None
        best_abs = -1.0

        for flow in flows:
            terminal = getattr(flow, "Terminal", None)
            terminal_id = _canonical_id(terminal)
            if not terminal_id:
                continue

            eq = network_index["terminals_to_equipment"].get(terminal_id)
            if eq != equipment_obj:
                continue

            p = getattr(flow, "p", 0.0)
            q = getattr(flow, "q", 0.0)

            if metric == "S":
                val = math.sqrt(p**2 + q**2)
            elif metric == "P":
                val = float(p)
            else:
                val = float(q)

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
            "metric": metric,
        }

        if metric == "S":
            row["apparent_power_MVA"] = float(best_val)
            row["rated_MVA"] = rated_mva
            row["loading_percent"] = (float(best_val) / rated_mva * 100.0) if (rated_mva is not None and rated_mva > 0) else None
        elif metric == "P":
            row["active_power_MW"] = float(best_val)
        else:
            row["reactive_power_MVAr"] = float(best_val)

        results.append(row)

    results.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"]))
    return results


def query_equipment_voltage_over_time(snapshot_cache, network_index, equipment_obj):
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
                "voltage_kV": float(v),
            })

    results.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"], r["terminal_id"]))
    return results


# =============================================================================
# Einzel-Equipment: Summaries
# =============================================================================

def summarize_metric(results):
    if not results:
        return {"type": None, "message": "Keine Daten verfügbar"}

    metric = results[0]["metric"]

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
        "num_datapoints": len(values),
    }

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
                "rated_MVA": results[0].get("rated_MVA", None),
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


def summarize_voltage(results):
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
        "num_datapoints": len(results),
    }


# =============================================================================
# Topologie Queries
# =============================================================================

def query_equipment_topology_neighbors(
    network_index,
    equipment_obj,
    level: str = "connectivity",
    allowed_neighbor_classes=None,
):
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
# Topologie Summaries
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


# =============================================================================
# Generische Mengen-Helper für Kombinationen
# =============================================================================

def get_component_equipment_objects(
    network_index,
    reference_equipment_obj,
    level: str = "topological",
    target_equipment_type: str | None = None,
):
    component_result = query_equipment_connected_component(
        network_index=network_index,
        equipment_obj=reference_equipment_obj,
        level=level,
    )

    equipment_nodes = component_result.get("equipment_nodes", []) or []
    if not equipment_nodes:
        return []

    obj_lookup = _build_equipment_object_lookup(network_index)

    out = []
    for node in equipment_nodes:
        if target_equipment_type and node.get("cim_class") != target_equipment_type:
            continue

        eq_id = node.get("equipment_id")
        eq_obj = obj_lookup.get(eq_id)
        if eq_obj is not None:
            out.append(eq_obj)

    return out


def get_neighbor_equipment_objects(
    network_index,
    reference_equipment_obj,
    level: str = "connectivity",
    target_equipment_type: str | None = None,
):
    neighbors = query_equipment_topology_neighbors(
        network_index=network_index,
        equipment_obj=reference_equipment_obj,
        level=level,
        allowed_neighbor_classes=None,
    )

    if not neighbors:
        return []

    obj_lookup = _build_equipment_object_lookup(network_index)

    out = []
    for nbr in neighbors:
        if target_equipment_type and nbr.get("neighbor_type") != target_equipment_type:
            continue

        eq_id = nbr.get("neighbor_equipment_id")
        eq_obj = obj_lookup.get(eq_id)
        if eq_obj is not None:
            out.append(eq_obj)

    return out


def aggregate_metric_over_equipment_set(
    snapshot_cache,
    network_index,
    equipment_objects,
    metric: str = "P",
    aggregation: str = "max",
):
    metric = (metric or "P").upper().strip()
    aggregation = (aggregation or "max").lower().strip()

    if metric == "P":
        value_key = "active_power_MW"
        unit = "MW"
    elif metric == "Q":
        value_key = "reactive_power_MVAr"
        unit = "MVAr"
    else:
        value_key = "apparent_power_MVA"
        unit = "MVA"

    all_rows = []

    for eq_obj in equipment_objects or []:
        rows = query_equipment_metric_over_time(
            snapshot_cache=snapshot_cache,
            network_index=network_index,
            equipment_obj=eq_obj,
            metric=metric,
        )
        all_rows.extend(rows)

    valid_rows = [r for r in all_rows if r.get(value_key) is not None]

    if not valid_rows:
        return {
            "type": "equipment_set_metric_aggregation",
            "found": False,
            "metric": metric,
            "aggregation": aggregation,
            "unit": unit,
            "num_equipment": len(equipment_objects or []),
            "num_rows": 0,
        }

    if aggregation == "min":
        best = min(valid_rows, key=lambda r: r[value_key])
    else:
        best = max(valid_rows, key=lambda r: r[value_key])

    return {
        "type": "equipment_set_metric_aggregation",
        "found": True,
        "metric": metric,
        "aggregation": aggregation,
        "unit": unit,
        "num_equipment": len(equipment_objects or []),
        "num_rows": len(valid_rows),
        "best_equipment": best.get("equipment"),
        "best_type": best.get("type"),
        "best_value": best.get(value_key),
        "best_timestamp": best.get("timestamp_str"),
        "best_snapshot": best.get("snapshot"),
    }