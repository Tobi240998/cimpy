from pathlib import Path
import cimpy
import os


def load_cim_folder(root_folder):
    """
    Lädt alle CIM-Fälle aus Unterordnern.
    
    Rückgabe:
        dict: {Fallname: {Objekttyp: DataFrame}}
    """
    cim_data = {}

    root_folder = Path(root_folder)

    # Alle entpackten Netz-Ordner sammeln
    case_dirs = sorted(p for p in root_folder.iterdir() if p.is_dir())
    print(f"Gefundene entpackte CIM-Fälle: {len(case_dirs)}")

    for case_dir in case_dirs:
        # ---------------------------------------------
        # XML-Dateien für diesen CIM-Fall sammeln
        # ---------------------------------------------
        xml_files = sorted(case_dir.glob("*.xml"))

        if not xml_files:
            print(f"Keine XML-Dateien in {case_dir.name}, überspringe...")
            continue

        print(f"Lade CIM-Fall: {case_dir.name} ({len(xml_files)} XML-Dateien)")

        # ---------------------------------------------
        # CIMpy-Import
        # ---------------------------------------------
        try:
            cim_case = cimpy.cim_import(
                [str(p) for p in xml_files],
                "cgmes_v2_4_15"
            )
            cim_data[case_dir.name] = cim_case

        except Exception as e:
            print(f"Fehler beim Import von {case_dir.name}: {e}")

    return cim_data

def load_cim_snapshots(root_folder):
    snapshots = {}

    for case in sorted(os.listdir(root_folder)):
        case_path = os.path.join(root_folder, case)
        if not os.path.isdir(case_path):
            continue

        xml_files = [
            os.path.join(case_path, f)
            for f in os.listdir(case_path)
            if f.lower().endswith(".xml")
        ]

        if not xml_files:
            continue

        snapshots[case] = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

    return snapshots