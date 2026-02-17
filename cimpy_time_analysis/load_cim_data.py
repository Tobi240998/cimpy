from pathlib import Path
import cimpy
from cim_object_utils import collect_all_cim_objects


def load_cim_snapshots(root_folder):
    """
    Lädt alle CIM-Fälle aus Unterordnern.
    Rückgabe:
        dict: {Fallname: cim_case}
    """

    snapshots = {}
    root_folder = Path(root_folder)
    case_dirs = sorted(p for p in root_folder.iterdir() if p.is_dir())

    for case_dir in case_dirs:

        xml_files = sorted(case_dir.glob("*.xml"))

        if not xml_files:
            print(f"Keine XML-Dateien in {case_dir.name}, überspringe...")
            continue

        try:
            cim_case = cimpy.cim_import(
                [str(p) for p in xml_files],
                "cgmes_v2_4_15"
            )
            snapshots[case_dir.name] = cim_case

        except Exception as e:
            print(f"Fehler beim Import von {case_dir.name}: {e}")

    return snapshots


def build_network_index(cim_snapshots):
    """
    Baut statischen Netzwerk-Index.
    """

    first_snapshot = next(iter(cim_snapshots.values()))
    all_objects = collect_all_cim_objects(first_snapshot)

    network_index = {
        "terminals_to_equipment": {},   # terminal_mRID → equipment_object
        "equipment_by_type": {}         # class_name → {mRID → object}
    }

    for obj in all_objects:

        cls_name = obj.__class__.__name__
        obj_id = getattr(obj, "mRID", None)

        if not obj_id:
            continue

        # Equipment sammeln
        if hasattr(obj, "Terminals"):
            network_index["equipment_by_type"].setdefault(cls_name, {})
            network_index["equipment_by_type"][cls_name][obj_id] = obj

        # Terminal → Equipment Mapping
        if cls_name == "Terminal":
            equipment = getattr(obj, "ConductingEquipment", None)
            if equipment:
                terminal_id = obj.mRID
                network_index["terminals_to_equipment"][terminal_id] = equipment

    return network_index
