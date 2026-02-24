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
    """
    Generische Metric-Query für jedes ConductingEquipment, das über Terminals SvPowerFlow-Werte besitzt.

    metric:
      - "S": Scheinleistung (sqrt(P^2 + Q^2))  -> Ergebnis-Key: apparent_power_MVA
      - "P": Wirkleistung (p)                  -> Ergebnis-Key: active_power_MW
      - "Q": Blindleistung (q)                 -> Ergebnis-Key: reactive_power_MVAr

    Aggregation über Terminals:
      - Wir wählen den Terminal-Eintrag mit größtem Betrag (abs), um Doppelzählung / Vorzeichen-Kompensation
        (z.B. bei Trafos HV/LV) zu vermeiden.
      - Für Verbraucher (meist 1 Terminal) ist das identisch mit "der Wert".
    """
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

        # Wir merken uns pro Snapshot den "besten" Terminal-Wert (größter Betrag)
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

            if metric == "S":
                val = math.sqrt(p**2 + q**2)
            elif metric == "P":
                val = float(p)
            else:  # metric == "Q"
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


def query_equipment_power_over_time(snapshot_cache, network_index, equipment_obj):  #bestimmt maximale Scheinleistung des gewählten Equipments
    # Backwards-Compatibility: alte Funktion bleibt und nutzt die generische Metric-Query für S
    return query_equipment_metric_over_time(snapshot_cache, network_index, equipment_obj, metric="S")


def query_equipment_voltage_over_time(snapshot_cache, network_index, equipment_obj):  #bestimmt Spannung des gewählten Equipments; gleicher Aufbau wie bei PowerFlow außer anderem Mapping und Ziehen der Spannung
    """
    FIX: Terminals werden NICHT aus equipment_obj.Terminals gezogen,
    sondern über network_index["equipment_to_terminal_ids"].

    Pfad:
      equipment_id -> terminal_ids -> connectivityNode_id -> topologicalNode_id -> SvVoltage.v

    Hinweis:
    - Diese Query ist typunabhängig: funktioniert für PowerTransformer UND EnergyConsumer,
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
    """
    Generische Zusammenfassung für Metrik-Zeitreihen:
    - S: apparent_power_MVA (mit optionaler Auslastung %)
    - P: active_power_MW
    - Q: reactive_power_MVAr
    """
    if not results:
        return {"type": None, "message": "Keine Daten verfügbar"}

    metric = results[0].get("metric", "S")

    if metric == "S":
        value_key = "apparent_power_MVA"
        unit = "MVA"
    elif metric == "P":
        value_key = "active_power_MW"
        unit = "MW"
    else:
        value_key = "reactive_power_MVAr"
        unit = "MVAr"

    values = [r[value_key] for r in results if value_key in r]  #bildet eine Reihe der Werte in zeitlich geordneter Reihenfolge
    if not values:
        return {"type": results[0].get("type"), "message": "Keine Werte verfügbar"}

    peak_entry = max(results, key=lambda r: r.get(value_key, float("-inf")))  #der Eintrag mit dem Maximalwert wird gespeichert
    min_entry = min(results, key=lambda r: r.get(value_key, float("inf")))    #der Eintrag mit dem Minimalwert wird gespeichert

    summary = {
        "type": results[0].get("type"),
        "metric": metric,
        "unit": unit,
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": sum(values) / len(values),
        "peak_value": peak_entry.get(value_key),
        "peak_snapshot": peak_entry.get("snapshot"),
        "peak_timestamp": peak_entry.get("timestamp_str"),
        "peak_equipment": peak_entry.get("equipment"),
        "min_value_at_time": min_entry.get(value_key),
        "min_snapshot": min_entry.get("snapshot"),
        "min_timestamp": min_entry.get("timestamp_str"),
        "min_equipment": min_entry.get("equipment"),
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


def summarize_powerflow(results):
    # Backwards-Compatibility: alte Funktion bleibt, nutzt generische Summary
    return summarize_metric(results)


def summarize_voltage(results):  #gleiche Vorgehensweise wie bei summarize_powerflow
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
