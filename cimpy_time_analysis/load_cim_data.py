from pathlib import Path
import cimpy
from cim_object_utils import collect_all_cim_objects
from asset_resolver import normalize_text


def load_cim_snapshots(root_folder):
    """
    Lädt alle CIM-Fälle aus Unterordnern.

    Rückgabe:
        dict: {Fallname: cim_case}
    """
    snapshots = {}
    root_folder = Path(root_folder)

    # Alle entpackten Netz-Ordner sammeln
    case_dirs = sorted(p for p in root_folder.iterdir() if p.is_dir())

    for case_dir in case_dirs:
        # ---------------------------------------------
        # XML-Dateien für diesen CIM-Fall sammeln
        # ---------------------------------------------
        xml_files = sorted(case_dir.glob("*.xml"))

        if not xml_files:
            print(f"Keine XML-Dateien in {case_dir.name}, überspringe...")
            continue

        # ---------------------------------------------
        # CIMpy-Import
        # ---------------------------------------------
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
    Baut einen statischen Netzwerk-Index mit:
    - equipment_by_id: mRID → Objekt
    - terminals_to_equipment: terminal_mRID → equipment_object
    - equipment_by_type: class_name → {mRID → object}
    - equipment_name_index: class_name → {normalized_name → object}
    """

    # Struktur einmal aus erstem Snapshot ziehen
    first_snapshot = next(iter(cim_snapshots.values()))
    all_objects = collect_all_cim_objects(first_snapshot)

    network_index = {
        "equipment_by_id": {},           # mRID → Objekt
        "terminals_to_equipment": {},    # terminal_mRID → equipment_object
        "equipment_by_type": {},         # class_name → {mRID → object}
        "equipment_name_index": {}       # class_name → {normalized_name → object}
    }

    for obj in all_objects:
        cls_name = obj.__class__.__name__
        obj_id = getattr(obj, "mRID", None)

        if not obj_id:
            continue

        # 1️⃣ Equipment sammeln: als "Equipment" betrachten wir alles mit Terminals
        # (PowerTransformer, ACLineSegment, EnergyConsumer, etc.)
        if hasattr(obj, "Terminals"):
            network_index["equipment_by_id"][obj_id] = obj

            network_index["equipment_by_type"].setdefault(cls_name, {})
            network_index["equipment_by_type"][cls_name][obj_id] = obj

            # Name Index (für robustes Matching aus User-Input)
            name = getattr(obj, "name", None)
            if name:
                norm_name = normalize_text(name)
                network_index["equipment_name_index"].setdefault(cls_name, {})
                # bei Kollisionen überschreibt der letzte — i.d.R. ok, sonst später auf Liste erweitern
                network_index["equipment_name_index"][cls_name][norm_name] = obj

        # 2️⃣ Terminal → Equipment Mapping
        if cls_name == "Terminal":
            equipment = getattr(obj, "ConductingEquipment", None)
            if equipment:
                equipment_id = getattr(equipment, "mRID", None)
                terminal_id = getattr(obj, "mRID", None)
                if equipment_id and terminal_id:
                    network_index["terminals_to_equipment"][terminal_id] = equipment

    return network_index
