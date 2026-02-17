import math


def query_equipment_power_over_time(snapshot_cache, network_index, equipment_obj):
    """
    Berechnet eine Zeitreihe der Scheinleistung (MVA) für EIN konkretes Equipment
    über alle Snapshots.

    Für Trafos: Maximum über die zugehörigen Terminals (HV/LV-Seite).
    """

    if equipment_obj is None:
        return []

    equipment_type = equipment_obj.__class__.__name__
    equipment_name = getattr(equipment_obj, "name", getattr(equipment_obj, "mRID", "UNKNOWN"))

    results = []

    for snapshot, data in snapshot_cache.items():

        flows = data.get("flows", [])
        if not flows:
            continue

        max_s = 0.0

        for flow in flows:
            terminal = getattr(flow, "Terminal", None)
            if not terminal:
                continue

            terminal_id = getattr(terminal, "mRID", None)
            if not terminal_id:
                continue

            eq = network_index["terminals_to_equipment"].get(terminal_id)
            if not eq:
                continue

            if eq != equipment_obj:
                continue

            p = getattr(flow, "p", 0.0)
            q = getattr(flow, "q", 0.0)
            s = math.sqrt(p**2 + q**2)

            max_s = max(max_s, s)

        # Nur wenn in diesem Snapshot überhaupt etwas gefunden wurde
        if max_s > 0:
            results.append({
                "snapshot": snapshot,
                "equipment": equipment_name,
                "type": equipment_type,
                "apparent_power_MVA": max_s
            })

    return results


def summarize_powerflow(results):
    """
    Strukturierte Summary für LLM/Agent.
    Erwartet results mit key: apparent_power_MVA
    """

    if not results:
        return {
            "type": None,
            "message": "Keine Daten verfügbar"
        }

    values = [r["apparent_power_MVA"] for r in results]
    peak_entry = max(results, key=lambda r: r["apparent_power_MVA"])
    min_entry = min(results, key=lambda r: r["apparent_power_MVA"])

    return {
        "type": results[0]["type"],
        "unit": "MVA",
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": sum(values) / len(values),
        "peak_snapshot": peak_entry["snapshot"],
        "peak_equipment": peak_entry["equipment"],
        "min_snapshot": min_entry["snapshot"],
        "min_equipment": min_entry["equipment"],
        "num_datapoints": len(results)
    }
