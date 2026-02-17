import math


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


def query_equipment_power_over_time(snapshot_cache, network_index, equipment_obj):
    if equipment_obj is None:
        return []

    equipment_type = equipment_obj.__class__.__name__
    equipment_name = getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN"))

    results = []

    for snapshot, data in snapshot_cache.items():
        flows = data.get("flows", [])
        if not flows:
            continue

        ts = data.get("timestamp", None)
        ts_str = data.get("timestamp_str", None)

        max_s = 0.0

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
            s = math.sqrt(p**2 + q**2)
            max_s = max(max_s, s)

        if max_s > 0:
            results.append({
                "snapshot": snapshot,
                "timestamp": ts,
                "timestamp_str": ts_str,
                "equipment": equipment_name,
                "type": equipment_type,
                "apparent_power_MVA": max_s
            })

    results.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"]))
    return results


def query_equipment_voltage_over_time(snapshot_cache, network_index, equipment_obj):
    """
    FIX: Terminals werden NICHT aus equipment_obj.Terminals gezogen,
    sondern über network_index["equipment_to_terminal_ids"].

    Pfad:
      equipment_id -> terminal_ids -> connectivityNode_id -> topologicalNode_id -> SvVoltage.v
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


def summarize_powerflow(results):
    if not results:
        return {"type": None, "message": "Keine Daten verfügbar"}

    values = [r["apparent_power_MVA"] for r in results]
    peak_entry = max(results, key=lambda r: r["apparent_power_MVA"])
    min_entry = min(results, key=lambda r: r["apparent_power_MVA"])

    return {
        "type": results[0]["type"],
        "unit": "MVA",
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": sum(values) / len(values),
        "peak_value": peak_entry["apparent_power_MVA"],
        "peak_snapshot": peak_entry["snapshot"],
        "peak_timestamp": peak_entry.get("timestamp_str"),
        "peak_equipment": peak_entry["equipment"],
        "min_value_at_time": min_entry["apparent_power_MVA"],
        "min_snapshot": min_entry["snapshot"],
        "min_timestamp": min_entry.get("timestamp_str"),
        "min_equipment": min_entry["equipment"],
        "num_datapoints": len(results)
    }


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
        "num_datapoints": len(results)
    }
