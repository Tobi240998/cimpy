from cim_object_utils import collect_all_cim_objects


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


def preprocess_snapshots(cim_snapshots):
    cache = {}

    for name, cim_result in cim_snapshots.items():
        all_objects = collect_all_cim_objects(cim_result) #Laden aller Objekte

        flows = [
            obj for obj in all_objects
            if obj.__class__.__name__ == "SvPowerFlow" #Speichern aller Leistungen
        ]

        voltages = [
            obj for obj in all_objects
            if obj.__class__.__name__ == "SvVoltage" #Speichern aller Spannungen
        ]

        # Mapping: TopologicalNode ID -> SvVoltage
        voltage_by_node = {}
        for sv in voltages:
            node = getattr(sv, "TopologicalNode", None) #Zuordnung des topological nodes zu Spannungen
            node_id = _canonical_id(node)
            if node_id:
                voltage_by_node[node_id] = sv

        scenario_time = cim_result.get("scenario_time", None) #Speichern des Zeitpunktes der Messung / Simulation

        cache[name] = {
            "flows": flows,
            "voltages": voltages,
            "voltage_by_node": voltage_by_node,
            "timestamp": scenario_time,
            "timestamp_str": scenario_time.isoformat() if scenario_time else None
        }

    return cache
