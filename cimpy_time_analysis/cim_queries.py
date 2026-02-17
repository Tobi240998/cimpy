import math


def query_equipment_power_over_time(snapshot_cache, network_index, equipment_type):
    """
    Berechnet die maximale Scheinleistung (MVA) pro Equipment und Snapshot.
    Für Trafos wird das Maximum der Wicklungen verwendet.
    """

    results = []

    for snapshot, data in snapshot_cache.items():

        flows = data["flows"]
        equipment_loading = {}

        for flow in flows:

            terminal = getattr(flow, "Terminal", None)
            if not terminal:
                continue

            terminal_id = getattr(terminal, "mRID", None)
            if not terminal_id:
                continue

            equipment = network_index["terminals_to_equipment"].get(terminal_id)
            if not equipment:
                continue

            if equipment.__class__.__name__ != equipment_type:
                continue

            # Scheinleistung berechnen
            p = getattr(flow, "p", 0.0)
            q = getattr(flow, "q", 0.0)
            s = math.sqrt(p**2 + q**2)

            name = getattr(equipment, "name", equipment.mRID)

            # Maximum über Terminals (wichtig für Trafos)
            equipment_loading[name] = max(
                equipment_loading.get(name, 0.0),
                s
            )

        for name, value in equipment_loading.items():
            results.append({
                "snapshot": snapshot,
                "equipment": name,
                "apparent_power_MVA": value,
                "type": equipment_type
            })

    return results


# -------------------------------------------------------
# Zusammenfassung für LLM-Ausgabe
# -------------------------------------------------------

def summarize_powerflow(results):
    """
    Erstellt eine strukturierte Zusammenfassung der
    Scheinleistungen über alle Snapshots hinweg.
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
