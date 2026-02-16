from cim_object_utils import collect_all_cim_objects

# Berechnet die im Netzmodell vorhandenen Leistungen. Muss noch angepasst werden, dass nur der gefragte Typ (z. B. Trafo) betrachtet wird.
def query_total_powerflow_over_time(snapshot_cache):
    results = []

    for snapshot, data in snapshot_cache.items():
        flows = data.get("SvPowerFlow", [])
        if not flows:
            continue

        total_p_mw = sum(obj.p for obj in flows)

        results.append({
            "snapshot": snapshot,
            "total_active_power": total_p_mw
        })

    return results


def query_trafo_power_over_time(snapshot_cache):
    results = []

    for snapshot, data in snapshot_cache.items():
        flows = data.get("flows", [])
        trafos = data.get("trafos", [])

        if not flows or not trafos:
            continue

        trafo_power = {t.name: 0.0 for t in trafos}

        for flow in flows:
            terminal = getattr(flow, "Terminal", None)
            if not terminal:
                continue

            terminals = terminal if isinstance(terminal, list) else [terminal]

            for t in terminals:
                equipment = getattr(t, "ConductingEquipment", None)
                if not equipment:
                    continue

                if equipment.__class__.__name__ == "PowerTransformer":
                    trafo_name = getattr(equipment, "name", "unknown")
                    trafo_power[trafo_name] += getattr(flow, "p", 0.0)

        for name, value in trafo_power.items():
            results.append({
                "snapshot": snapshot,
                "trafo": name,
                "total_active_power_MW": value
            })

    return results






def summarize_powerflow(results):
    """
    Generische Zusammenfassung von PowerFlow-Ergebnissen.
    
    Funktioniert für:
    1. query_total_powerflow_over_time (gesamt Netz)
    2. query_trafo_power_over_time (pro Trafo)
    
    Ergebnis:
        dict: Zusammenfassung der min/max/mean und Peak-Snapshot,
              optional pro Trafo, falls vorhanden.
    """
    from collections import defaultdict

    # Prüfen, ob es Trafo-Daten sind
    is_trafo = "trafo" in results[0] if results else False

    if is_trafo:
        # pro Trafo zusammenfassen
        summary = {}
        trafos = defaultdict(list)

        for r in results:
            trafos[r["trafo"]].append(r)

        for trafo, entries in trafos.items():
            values = [e["total_active_power_MW"] for e in entries]
            peak_snapshot = max(entries, key=lambda e: e["total_active_power_MW"])["snapshot"]
            rated = entries[0].get("rated_MVA", None)

            summary[trafo] = {
                "min_MW": min(values),
                "max_MW": max(values),
                "mean_MW": sum(values)/len(values),
                "peak_snapshot": peak_snapshot,
                "rated_MVA": rated
            }

        return summary

    else:
        # Gesamt-Netz-Zusammenfassung
        summary = {
            "min_MW": min(values),
            "max_MW": max(values),
            "mean_MW": sum(values)/len(values),
            "peak_snapshot": peak_snapshot,
            "type": "network_total"
        }
        return summary

