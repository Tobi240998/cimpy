from cimpy.cimpy_time_analysis.cim_object_utils import collect_all_cim_objects


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


# =============================================================================
# Interne Helper
# =============================================================================

def _collect_objects_by_class(all_objects, class_names):
    class_names = set(class_names or [])
    if not class_names:
        return {}

    grouped = {cls_name: [] for cls_name in class_names}

    for obj in all_objects:
        cls_name = obj.__class__.__name__
        if cls_name in grouped:
            grouped[cls_name].append(obj)

    return grouped


def _build_voltage_by_node(voltages):
    """
    Mapping: TopologicalNode ID -> SvVoltage
    """
    voltage_by_node = {}

    for sv in voltages or []:
        node = getattr(sv, "TopologicalNode", None)
        node_id = _canonical_id(node)
        if node_id:
            voltage_by_node[node_id] = sv

    return voltage_by_node


def _build_flow_by_terminal(flows):
    """
    Optionales Zusatz-Mapping für spätere gezielte Queries:
    Terminal ID -> Liste[SvPowerFlow]
    """
    flow_by_terminal = {}

    for flow in flows or []:
        terminal = getattr(flow, "Terminal", None)
        terminal_id = _canonical_id(terminal)
        if not terminal_id:
            continue

        flow_by_terminal.setdefault(terminal_id, []).append(flow)

    return flow_by_terminal


def _normalize_state_request(state_types=None):
    """
    Normalisiert gewünschte State-Typen auf eine Menge von CIM-Klassennamen.

    Akzeptiert z.B.:
    - None
    - "SvVoltage"
    - "SvPowerFlow"
    - ["SvVoltage", "SvPowerFlow"]
    """
    if state_types is None:
        return {"SvPowerFlow", "SvVoltage"}

    if isinstance(state_types, str):
        state_types = [state_types]

    normalized = set()
    for s in state_types:
        if not s:
            continue
        normalized.add(str(s).strip())

    return normalized


def _build_snapshot_cache_entry(cim_result, requested_state_types):
    """
    Baut einen Cache-Eintrag für genau einen Snapshot.

    Rückgabe ist absichtlich stabil und tool-freundlich:
    - immer timestamp / timestamp_str
    - State-Felder nur dann befüllt, wenn angefordert
    """
    all_objects = collect_all_cim_objects(cim_result)

    grouped = _collect_objects_by_class(all_objects, requested_state_types)

    scenario_time = cim_result.get("scenario_time", None)

    entry = {
        "timestamp": scenario_time,
        "timestamp_str": scenario_time.isoformat() if scenario_time else None,
        "scenario_time": cim_result.get("scenario_time", None),
        "scenario_time_str": cim_result.get("scenario_time_str", None),
        "scenario_time_source": cim_result.get("scenario_time_source", None),
        "snapshot_name": cim_result.get("snapshot_name", None),
        "requested_state_types": sorted(requested_state_types),
        "state_counts": {},
    }

    # Rückwärtskompatibilität: bisher erwartete Keys
    if "SvPowerFlow" in requested_state_types:
        flows = grouped.get("SvPowerFlow", [])
        entry["flows"] = flows
        entry["flow_by_terminal"] = _build_flow_by_terminal(flows)
        entry["state_counts"]["SvPowerFlow"] = len(flows)
    else:
        entry["flows"] = []
        entry["flow_by_terminal"] = {}
        entry["state_counts"]["SvPowerFlow"] = 0

    if "SvVoltage" in requested_state_types:
        voltages = grouped.get("SvVoltage", [])
        entry["voltages"] = voltages
        entry["voltage_by_node"] = _build_voltage_by_node(voltages)
        entry["state_counts"]["SvVoltage"] = len(voltages)
    else:
        entry["voltages"] = []
        entry["voltage_by_node"] = {}
        entry["state_counts"]["SvVoltage"] = 0

    # Generische Ablage für spätere Erweiterungen
    entry["states_by_class"] = grouped

    return entry


# =============================================================================
# Neue, selektive API
# =============================================================================

def preprocess_snapshots_for_states(cim_snapshots, state_types=None):
    """
    Preprocess nur für die gewünschten State-Typen.

    Beispiele:
    - preprocess_snapshots_for_states(cim_snapshots, ["SvVoltage"])
    - preprocess_snapshots_for_states(cim_snapshots, ["SvPowerFlow"])
    - preprocess_snapshots_for_states(cim_snapshots, ["SvVoltage", "SvPowerFlow"])

    Rückgabe:
    {
        "snapshot_name": {
            "timestamp": ...,
            "timestamp_str": ...,
            "flows": [...],            # nur falls SvPowerFlow angefordert
            "voltages": [...],         # nur falls SvVoltage angefordert
            "voltage_by_node": {...},  # nur falls SvVoltage angefordert
            "states_by_class": {...},
            ...
        }
    }
    """
    requested_state_types = _normalize_state_request(state_types)
    cache = {}

    for name, cim_result in cim_snapshots.items():
        entry = _build_snapshot_cache_entry(
            cim_result=cim_result,
            requested_state_types=requested_state_types,
        )

        # Snapshot-Name im Cache-Key und im Eintrag sauber konsistent halten
        if not entry.get("snapshot_name"):
            entry["snapshot_name"] = name

        cache[name] = entry

    return cache


def preprocess_snapshots_for_state(cim_snapshots, state_type):
    """
    Komfortfunktion für genau einen State-Typ.
    """
    return preprocess_snapshots_for_states(cim_snapshots, [state_type])


def preprocess_voltage_snapshots(cim_snapshots):
    """
    Komfortfunktion: nur SvVoltage.
    """
    return preprocess_snapshots_for_states(cim_snapshots, ["SvVoltage"])


def preprocess_powerflow_snapshots(cim_snapshots):
    """
    Komfortfunktion: nur SvPowerFlow.
    """
    return preprocess_snapshots_for_states(cim_snapshots, ["SvPowerFlow"])


# =============================================================================
# MCP-/Tool-freundliche Metadata-Helfer
# =============================================================================

def summarize_snapshot_cache(snapshot_cache):
    """
    Liefert eine kompakte Übersicht über den erzeugten Cache.
    Gut für Debugging und spätere Tool-Ausgaben.
    """
    if not snapshot_cache:
        return {
            "num_snapshots": 0,
            "requested_state_types": [],
            "snapshot_names": [],
            "total_state_counts": {},
        }

    snapshot_names = list(snapshot_cache.keys())

    requested_state_types = set()
    total_state_counts = {}

    for _, entry in snapshot_cache.items():
        for state_type in entry.get("requested_state_types", []):
            requested_state_types.add(state_type)

        for state_type, count in entry.get("state_counts", {}).items():
            total_state_counts[state_type] = total_state_counts.get(state_type, 0) + int(count or 0)

    return {
        "num_snapshots": len(snapshot_cache),
        "requested_state_types": sorted(requested_state_types),
        "snapshot_names": snapshot_names,
        "total_state_counts": total_state_counts,
    }


def filter_snapshot_cache_by_available_state(snapshot_cache, required_state_type):
    """
    Filtert den Cache auf Snapshots, in denen ein bestimmter State-Type
    tatsächlich vorhanden ist.
    """
    out = {}

    for name, entry in (snapshot_cache or {}).items():
        count = entry.get("state_counts", {}).get(required_state_type, 0)
        if count and count > 0:
            out[name] = entry

    return out


# =============================================================================
# Rückwärtskompatible Alt-API
# =============================================================================

def preprocess_snapshots(cim_snapshots):
    """
    Rückwärtskompatibel:
    verhält sich wie bisher und lädt standardmäßig SvPowerFlow + SvVoltage.

    Bestehender Code, der diese Funktion nutzt, muss dadurch nicht sofort geändert werden.
    """
    return preprocess_snapshots_for_states(
        cim_snapshots=cim_snapshots,
        state_types=["SvPowerFlow", "SvVoltage"],
    )