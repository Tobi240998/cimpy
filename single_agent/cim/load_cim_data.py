from pathlib import Path
from cimpy.cimpy import cim_import
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


from cimpy.cimpy_time_analysis.cim_object_utils import collect_all_cim_objects
from cimpy.cimpy_time_analysis.asset_resolver import normalize_text
from cimpy.cimpy_time_analysis.cim_topology_graph import build_cim_topology_graph, summarize_graph_basic


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


def _parse_cgmes_datetime(value: str):
    """
    Erwartet z.B. '2026-01-09T14:15:00Z' oder ISO-Strings.
    Gibt timezone-aware datetime (UTC) zurück, falls möglich.
    """
    if not value:
        return None

    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _extract_profile_and_times(xml_path: Path):
    """
    Liest md:Model.profile, md:Model.scenarioTime und md:Model.created aus einer CGMES XML-Datei.
    Gibt (profile_url, scenario_dt, created_dt) zurück.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None, None, None

    ns = {"md": "http://iec.ch/TC57/61970-552/ModelDescription/1#"}
    full_model = root.find(".//md:FullModel", ns)
    if full_model is None:
        return None, None, None

    profile = full_model.findtext(".//md:Model.profile", default=None, namespaces=ns)
    scen = full_model.findtext(".//md:Model.scenarioTime", default=None, namespaces=ns)
    created = full_model.findtext(".//md:Model.created", default=None, namespaces=ns)

    scenario_dt = _parse_cgmes_datetime(scen) if scen else None
    created_dt = _parse_cgmes_datetime(created) if created else None
    return profile, scenario_dt, created_dt


def _classify_profile(profile_url: str):
    """
    Grobe Klassifikation über URL-Substring.
    Wir machen das bewusst tolerant, weil Profile-URLs je nach Export variieren.
    """
    if not profile_url:
        return None

    p = profile_url.lower()
    if "statevariables" in p:
        return "SV"
    if "steadystatehypothesis" in p:
        return "SSH"
    if "topology" in p:
        return "TP"
    if "equipment" in p:
        return "EQ"
    return "OTHER"


def _extract_times_by_profile(xml_files):
    """
    Liest alle XMLs eines Snapshot-Ordners und gruppiert scenarioTime/created nach Profil (SV/SSH/TP/EQ).
    Zusätzlich: liefert eine 'default' scenario_time, die bevorzugt aus SV kommt.
    """
    times_by_profile = {
        "SV": {"scenario_time": None, "created": None, "files": []},
        "SSH": {"scenario_time": None, "created": None, "files": []},
        "TP": {"scenario_time": None, "created": None, "files": []},
        "EQ": {"scenario_time": None, "created": None, "files": []},
        "OTHER": {"scenario_time": None, "created": None, "files": []},
        None: {"scenario_time": None, "created": None, "files": []},
    }

    all_scenario_times = set()

    for f in xml_files:
        profile_url, scenario_dt, created_dt = _extract_profile_and_times(Path(f))
        prof = _classify_profile(profile_url)
        bucket = times_by_profile.get(prof, times_by_profile["OTHER"])

        bucket["files"].append(str(f))

        if scenario_dt and (bucket["scenario_time"] is None or scenario_dt > bucket["scenario_time"]):
            bucket["scenario_time"] = scenario_dt
        if created_dt and (bucket["created"] is None or created_dt > bucket["created"]):
            bucket["created"] = created_dt

        if scenario_dt:
            all_scenario_times.add(scenario_dt.isoformat())

    default_time = times_by_profile["SV"]["scenario_time"]
    default_source = "SV:md:Model.scenarioTime"

    if default_time is None:
        for key in ["SSH", "TP", "EQ", "OTHER", None]:
            dt = times_by_profile[key]["scenario_time"]
            if dt:
                default_time = dt
                default_source = f"{key}:md:Model.scenarioTime"
                break

    if default_time is None:
        dt = times_by_profile["SV"]["created"]
        if dt:
            default_time = dt
            default_source = "SV:md:Model.created"
        else:
            for key in ["SSH", "TP", "EQ", "OTHER", None]:
                dt = times_by_profile[key]["created"]
                if dt:
                    default_time = dt
                    default_source = f"{key}:md:Model.created"
                    break

    return times_by_profile, default_time, default_source, sorted(all_scenario_times)


# =============================================================================
# NEU: Discovery / Inventory (MCP-freundlich, leichtgewichtig)
# =============================================================================

def scan_snapshot_inventory(root_folder):
    """
    Scannt nur Metadaten der Snapshot-Ordner, ohne cim_import auszuführen.

    Rückgabe ist bewusst JSON-/Tool-freundlich:
    {
        "root_folder": "...",
        "snapshots": [
            {
                "snapshot_name": "...",
                "case_dir": "...",
                "xml_files": [...],
                "default_time": datetime|None,
                "default_time_str": "...",
                "default_time_source": "...",
                "times_by_profile": {...},
                "all_scenario_times": [...],
                "has_sv_profile": bool,
                "has_tp_profile": bool,
                "has_eq_profile": bool,
                "has_ssh_profile": bool,
            },
            ...
        ]
    }
    """
    root_folder = Path(root_folder)
    snapshots = []

    if not root_folder.exists():
        return {
            "root_folder": str(root_folder),
            "snapshots": [],
        }

    case_dirs = sorted(p for p in root_folder.iterdir() if p.is_dir())

    for case_dir in case_dirs:
        xml_files = sorted(case_dir.glob("*.xml"))
        if not xml_files:
            continue

        xml_files_str = [str(p) for p in xml_files]
        times_by_profile, default_time, default_source, all_scenario_times = _extract_times_by_profile(xml_files_str)

        snapshot_meta = {
            "snapshot_name": case_dir.name,
            "case_dir": str(case_dir),
            "xml_files": xml_files_str,
            "default_time": default_time,
            "default_time_str": default_time.isoformat() if default_time else None,
            "default_time_source": default_source,
            "times_by_profile": times_by_profile,
            "all_scenario_times": all_scenario_times,
            "has_sv_profile": bool(times_by_profile["SV"]["files"]),
            "has_tp_profile": bool(times_by_profile["TP"]["files"]),
            "has_eq_profile": bool(times_by_profile["EQ"]["files"]),
            "has_ssh_profile": bool(times_by_profile["SSH"]["files"]),
        }

        snapshots.append(snapshot_meta)

    snapshots.sort(key=lambda x: (
        x["default_time"] is None,
        x["default_time"] if x["default_time"] is not None else datetime.max.replace(tzinfo=timezone.utc),
        x["snapshot_name"],
    ))

    return {
        "root_folder": str(root_folder),
        "snapshots": snapshots,
    }


def _parse_iso_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def select_snapshot_names_by_time(snapshot_inventory, start_time=None, end_time=None):
    """
    Wählt Snapshot-Namen anhand des Zeitfensters [start_time, end_time) aus.

    Wenn start/end fehlen, werden alle Snapshot-Namen zurückgegeben.
    """
    snapshots = snapshot_inventory.get("snapshots", []) if snapshot_inventory else []

    start_dt = _parse_iso_datetime(start_time)
    end_dt = _parse_iso_datetime(end_time)

    if start_dt is None or end_dt is None:
        return [s["snapshot_name"] for s in snapshots]

    selected = []
    for meta in snapshots:
        ts = meta.get("default_time")
        if ts is None:
            continue
        if start_dt <= ts < end_dt:
            selected.append(meta["snapshot_name"])

    return selected


def get_snapshot_metadata_by_name(snapshot_inventory, snapshot_name):
    snapshots = snapshot_inventory.get("snapshots", []) if snapshot_inventory else []
    for meta in snapshots:
        if meta.get("snapshot_name") == snapshot_name:
            return meta
    return None


# =============================================================================
# NEU: Gezieltes Laden einzelner / ausgewählter Snapshots
# =============================================================================

def load_single_snapshot_from_metadata(snapshot_meta):
    """
    Lädt genau einen Snapshot vollständig per cim_import.
    Erwartet einen Eintrag aus scan_snapshot_inventory(...).
    """
    if not snapshot_meta:
        return None

    xml_files_str = snapshot_meta.get("xml_files", [])
    if not xml_files_str:
        return None

    cim_case = cim_import(xml_files_str, "cgmes_v2_4_15")

    default_time = snapshot_meta.get("default_time")
    default_source = snapshot_meta.get("default_time_source")
    times_by_profile = snapshot_meta.get("times_by_profile", {})

    cim_case["scenario_time"] = default_time
    cim_case["scenario_time_source"] = default_source
    cim_case["scenario_time_str"] = default_time.isoformat() if default_time else None
    cim_case["profile_times"] = times_by_profile
    cim_case["snapshot_name"] = snapshot_meta.get("snapshot_name")
    cim_case["case_dir"] = snapshot_meta.get("case_dir")

    return cim_case


def load_single_snapshot(case_dir):
    """
    Komfortfunktion für einen einzelnen Snapshot-Ordner.
    """
    case_dir = Path(case_dir)
    inventory = scan_snapshot_inventory(case_dir.parent)
    snapshot_meta = get_snapshot_metadata_by_name(inventory, case_dir.name)
    if snapshot_meta is None:
        return None
    return load_single_snapshot_from_metadata(snapshot_meta)


def load_cim_snapshots_from_inventory(snapshot_inventory, selected_snapshot_names=None):
    """
    Lädt nur die ausgewählten Snapshots.
    Wenn selected_snapshot_names=None, werden alle Snapshots geladen.

    Rückgabe:
    {
        "snapshot_name": cim_case,
        ...
    }
    """
    snapshots = {}
    all_meta = snapshot_inventory.get("snapshots", []) if snapshot_inventory else []

    selected_set = set(selected_snapshot_names) if selected_snapshot_names is not None else None

    for snapshot_meta in all_meta:
        snapshot_name = snapshot_meta.get("snapshot_name")
        if selected_set is not None and snapshot_name not in selected_set:
            continue

        try:
            cim_case = load_single_snapshot_from_metadata(snapshot_meta)
            if cim_case is not None:
                snapshots[snapshot_name] = cim_case
        except Exception as e:
            print(f"Fehler beim Import von {snapshot_name}: {e}")

    return snapshots


def load_cim_snapshots(root_folder, selected_snapshot_names=None):
    """
    Rückwärtskompatible Funktion.
    Lädt standardmäßig alle Snapshots, optional nur eine Auswahl.
    """
    inventory = scan_snapshot_inventory(root_folder)
    return load_cim_snapshots_from_inventory(
        snapshot_inventory=inventory,
        selected_snapshot_names=selected_snapshot_names,
    )


def load_snapshots_for_time_window(root_folder, start_time=None, end_time=None, snapshot_inventory=None):
    """
    Lädt nur die Snapshots, deren default_time im gewünschten Zeitfenster liegt.
    """
    if snapshot_inventory is None:
        snapshot_inventory = scan_snapshot_inventory(root_folder)

    selected_names = select_snapshot_names_by_time(
        snapshot_inventory=snapshot_inventory,
        start_time=start_time,
        end_time=end_time,
    )

    return load_cim_snapshots_from_inventory(
        snapshot_inventory=snapshot_inventory,
        selected_snapshot_names=selected_names,
    )


def choose_base_snapshot_metadata(snapshot_inventory, preferred_snapshot_name=None):
    """
    Wählt den Basissnapshot für statischen Netzindex / Topologie.

    Priorität:
    1) preferred_snapshot_name, falls vorhanden
    2) erster Snapshot mit EQ+TP
    3) sonst erster Snapshot aus Inventory
    """
    snapshots = snapshot_inventory.get("snapshots", []) if snapshot_inventory else []
    if not snapshots:
        return None

    if preferred_snapshot_name:
        for meta in snapshots:
            if meta.get("snapshot_name") == preferred_snapshot_name:
                return meta

    for meta in snapshots:
        if meta.get("has_eq_profile") and meta.get("has_tp_profile"):
            return meta

    return snapshots[0]


def load_base_snapshot(root_folder, snapshot_inventory=None, preferred_snapshot_name=None):
    """
    Lädt genau einen Basissnapshot für Netzwerkindex und Topologie.
    """
    if snapshot_inventory is None:
        snapshot_inventory = scan_snapshot_inventory(root_folder)

    base_meta = choose_base_snapshot_metadata(
        snapshot_inventory=snapshot_inventory,
        preferred_snapshot_name=preferred_snapshot_name,
    )
    if base_meta is None:
        return None

    return load_single_snapshot_from_metadata(base_meta)


# =============================================================================
# Netzwerkindex: bevorzugt aus genau EINEM Basissnapshot
# =============================================================================

def build_network_index_from_snapshot(base_snapshot):
    """
    Baut den Netzwerkindex aus genau einem Snapshot.
    Das ist die saubere Ziel-API für späteres Tool-/MCP-Design.
    """
    if base_snapshot is None:
        return {
            "equipment_name_index": {},
            "terminals_to_equipment": {},
            "equipment_to_terminal_ids": {},
            "terminal_to_connectivitynode": {},
            "connectivitynode_to_topologicalnode": {},
            "topology_graph": None,
            "topology_graph_summary": {},
            "topology_graph_connectivity": None,
            "topology_graph_connectivity_summary": {},
            "topology_graph_topological": None,
            "topology_graph_topological_summary": {},
            "index_source_snapshot": None,
            "index_source_time": None,
            "index_source_time_str": None,
        }

    all_objects = collect_all_cim_objects(base_snapshot)

    network_index = {
        "equipment_name_index": {},

        # Terminal Mapping
        "terminals_to_equipment": {},
        "equipment_to_terminal_ids": {},

        # Voltage / Topology Mapping
        "terminal_to_connectivitynode": {},
        "connectivitynode_to_topologicalnode": {},

        # Metadaten für spätere Tool-/MCP-Nutzung
        "index_source_snapshot": base_snapshot.get("snapshot_name"),
        "index_source_time": base_snapshot.get("scenario_time"),
        "index_source_time_str": base_snapshot.get("scenario_time_str"),
    }

    # 1) Equipment-Name-Index
    for obj in all_objects:
        if not hasattr(obj, "mRID"):
            continue
        if not hasattr(obj, "name"):
            continue

        cls_name = obj.__class__.__name__
        name = getattr(obj, "name", None)
        if not name:
            continue

        norm = normalize_text(name)
        network_index["equipment_name_index"].setdefault(cls_name, {})
        network_index["equipment_name_index"][cls_name][norm] = obj

    # 2) Terminals sammeln
    for obj in all_objects:
        if obj.__class__.__name__ != "Terminal":
            continue

        terminal_id = _canonical_id(getattr(obj, "mRID", None))
        if not terminal_id:
            continue

        equipment = getattr(obj, "ConductingEquipment", None)
        if equipment is not None:
            network_index["terminals_to_equipment"][terminal_id] = equipment

            equipment_id = _canonical_id(equipment)
            if equipment_id:
                network_index["equipment_to_terminal_ids"].setdefault(equipment_id, [])
                network_index["equipment_to_terminal_ids"][equipment_id].append(terminal_id)

        cn = getattr(obj, "ConnectivityNode", None)
        cn_id = _canonical_id(cn)
        if cn_id:
            network_index["terminal_to_connectivitynode"][terminal_id] = cn_id

    # 3) ConnectivityNode -> TopologicalNode
    for obj in all_objects:
        cls = obj.__class__.__name__

        if cls == "ConnectivityNode":
            cn_id = _canonical_id(getattr(obj, "mRID", None))
            tn_id = _canonical_id(getattr(obj, "TopologicalNode", None))
            if cn_id and tn_id:
                network_index["connectivitynode_to_topologicalnode"][cn_id] = tn_id

        if cls == "TopologicalNode":
            tn_id = _canonical_id(getattr(obj, "mRID", None))
            cns = getattr(obj, "ConnectivityNodes", None)
            if tn_id and isinstance(cns, list):
                for cn in cns:
                    cn_id = _canonical_id(cn)
                    if cn_id:
                        network_index["connectivitynode_to_topologicalnode"][cn_id] = tn_id

    # 4) Topologiegraphen
    connectivity_graph = build_cim_topology_graph(
        first_snapshot=base_snapshot,
        network_index=network_index,
        level="connectivity",
        include_topology_nodes=False,
    )

    topological_graph = build_cim_topology_graph(
        first_snapshot=base_snapshot,
        network_index=network_index,
        level="topological",
        include_topology_nodes=False,
    )

    network_index["topology_graph"] = connectivity_graph
    network_index["topology_graph_summary"] = summarize_graph_basic(connectivity_graph)

    network_index["topology_graph_connectivity"] = connectivity_graph
    network_index["topology_graph_connectivity_summary"] = summarize_graph_basic(connectivity_graph)

    network_index["topology_graph_topological"] = topological_graph
    network_index["topology_graph_topological_summary"] = summarize_graph_basic(topological_graph)

    return network_index


def build_network_index(cim_snapshots):
    """
    Rückwärtskompatible Hülle:
    nimmt wie bisher ein Dict geladener Snapshots
    und baut den Index aus dem ersten Snapshot.
    """
    if not cim_snapshots:
        return build_network_index_from_snapshot(None)

    first_snapshot = next(iter(cim_snapshots.values()))
    return build_network_index_from_snapshot(first_snapshot)