from pathlib import Path
import cimpy
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

from cim_object_utils import collect_all_cim_objects
from asset_resolver import normalize_text


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

    # Falls ohne tzinfo, als UTC interpretieren
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

        # pro Profil jeweils den "jüngsten" Zeitpunkt merken (meist identisch; robust falls nicht)
        if scenario_dt and (bucket["scenario_time"] is None or scenario_dt > bucket["scenario_time"]):
            bucket["scenario_time"] = scenario_dt
        if created_dt and (bucket["created"] is None or created_dt > bucket["created"]):
            bucket["created"] = created_dt

        if scenario_dt:
            all_scenario_times.add(scenario_dt.isoformat())

    # Default: SV scenarioTime, sonst fallback auf irgendeinen scenarioTime, sonst created
    default_time = times_by_profile["SV"]["scenario_time"]
    default_source = "SV:md:Model.scenarioTime"

    if default_time is None:
        # irgendein vorhandenes scenarioTime (z.B. SSH/TP/EQ)
        for key in ["SSH", "TP", "EQ", "OTHER", None]:
            dt = times_by_profile[key]["scenario_time"]
            if dt:
                default_time = dt
                default_source = f"{key}:md:Model.scenarioTime"
                break

    if default_time is None:
        # fallback: created (SV bevorzugt)
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


def load_cim_snapshots(root_folder):
    snapshots = {}
    root_folder = Path(root_folder)
    case_dirs = sorted(p for p in root_folder.iterdir() if p.is_dir())

    # XML-Files sammeln
    for case_dir in case_dirs:
        xml_files = sorted(case_dir.glob("*.xml"))
        if not xml_files:
            print(f"Keine XML-Dateien in {case_dir.name}, überspringe...")
            continue
            #CIMpy-Import durchführen der gesammelten Files

        xml_files_str = [str(p) for p in xml_files]

        # Zeitstempel aus CGMES XML Header ziehen
        # - Wir halten Zeiten pro Profil (SV/SSH/TP/EQ) separat
        # - Für Spannung/Leistung nutzen wir später den SV-Zeitstempel als Default
        times_by_profile, default_time, default_source, all_scenario_times = _extract_times_by_profile(xml_files_str)

        # Optionaler Konsistenzcheck: wenn mehrere scenarioTimes im Ordner vorkommen, warnen
        if len(all_scenario_times) > 1:
            print(f"Warnung: Mehrere scenarioTime-Werte in {case_dir.name}: {all_scenario_times}")

        try:
            cim_case = cimpy.cim_import(xml_files_str, "cgmes_v2_4_15")

            # Default Zeitinfo (für SV-basierte Queries)
            cim_case["scenario_time"] = default_time
            cim_case["scenario_time_source"] = default_source
            cim_case["scenario_time_str"] = default_time.isoformat() if default_time else None

            # Profil-spezifische Zeiten (für spätere Erweiterungen: Schalter/Topologie)
            cim_case["profile_times"] = times_by_profile

            snapshots[case_dir.name] = cim_case

        except Exception as e:
            print(f"Fehler beim Import von {case_dir.name}: {e}")

    return snapshots


def build_network_index(cim_snapshots):
    """
    Index aus erstem Snapshot, wird dann für alle Snapshots genutzt.

    WICHTIGER Fix:
    Terminals eines Equipments werden NICHT über equipment.Terminals bestimmt,
    sondern über Terminal.ConductingEquipment (robust gegen 'leere' Terminals).
    """

    first_snapshot = next(iter(cim_snapshots.values())) #erster Snapshot aus dem Dict wird genommen
    all_objects = collect_all_cim_objects(first_snapshot) #Sammeln aller Objekte des ersten Snapshots ohne verschachtelte Container

    network_index = {
        "equipment_name_index": {},

        # Terminal Mapping
        "terminals_to_equipment": {},            # terminal_id -> equipment_object
        "equipment_to_terminal_ids": {},         # equipment_id -> [terminal_id, ...]

        # Voltage Mapping
        "terminal_to_connectivitynode": {},      # terminal_id -> connectivityNode_id
        "connectivitynode_to_topologicalnode": {}# connectivityNode_id -> topologicalNode_id
    }

    # 1) Equipment-Name-Index (für Resolver)
    for obj in all_objects:
        #Filtern aller Objekte, die Namen und mRID haben
        if not hasattr(obj, "mRID"):
            continue
        if not hasattr(obj, "name"):
            continue

        cls_name = obj.__class__.__name__
        name = getattr(obj, "name", None)
        if not name:
            continue

        norm = normalize_text(name) #normalisiert Namen zur besseren Erkennung (alles klein, ohne Leerzeichen, ...)
        network_index["equipment_name_index"].setdefault(cls_name, {})
        network_index["equipment_name_index"][cls_name][norm] = obj #Klassenkey, Vorbereitung für später wenn auch Lasten, Leitungen etc. gemappt werden

    # 2) Terminals sammeln (robuster Pfad)
    for obj in all_objects:
        if obj.__class__.__name__ != "Terminal":
            continue

        terminal_id = _canonical_id(getattr(obj, "mRID", None)) #Normalisieren der ID auf einheitliches Format
        if not terminal_id:
            continue

        equipment = getattr(obj, "ConductingEquipment", None)
        if equipment is not None:
            network_index["terminals_to_equipment"][terminal_id] = equipment #Zuordnung des Equipments zum Terminal

            equipment_id = _canonical_id(equipment)
            if equipment_id:
                network_index["equipment_to_terminal_ids"].setdefault(equipment_id, [])
                network_index["equipment_to_terminal_ids"][equipment_id].append(terminal_id) #Zuordnung der Terminal ID zum Equipment

        cn = getattr(obj, "ConnectivityNode", None)
        cn_id = _canonical_id(cn)
        if cn_id:
            network_index["terminal_to_connectivitynode"][terminal_id] = cn_id #Zuordnung Terminal zu Connectivity Node

    # 3) ConnectivityNode -> TopologicalNode
    for obj in all_objects:
        cls = obj.__class__.__name__

        if cls == "ConnectivityNode":
            cn_id = _canonical_id(getattr(obj, "mRID", None))
            tn_id = _canonical_id(getattr(obj, "TopologicalNode", None))
            if cn_id and tn_id:
                network_index["connectivitynode_to_topologicalnode"][cn_id] = tn_id #Zuordnung Connectivity Node und Topological Node

        if cls == "TopologicalNode":
            tn_id = _canonical_id(getattr(obj, "mRID", None))
            cns = getattr(obj, "ConnectivityNodes", None)
            if tn_id and isinstance(cns, list):
                for cn in cns:
                    cn_id = _canonical_id(cn)
                    if cn_id:
                        network_index["connectivitynode_to_topologicalnode"][cn_id] = tn_id #Zuordnung Connectivity Node und Topological Node

    return network_index
