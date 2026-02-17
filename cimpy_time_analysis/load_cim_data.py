from pathlib import Path
import cimpy
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
        try:
            cim_case = cimpy.cim_import([str(p) for p in xml_files], "cgmes_v2_4_15")
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
