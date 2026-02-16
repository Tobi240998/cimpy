from pathlib import Path
import cimpy


def load_cim_snapshots(root_folder):
    """
    Lädt alle CIM-Fälle aus Unterordnern.
    
    Rückgabe:
        dict: {Fallname: {Objekttyp: DataFrame}}
    """
    # Erstellung dict
    snapshots = {}

    # Umwandlung root_folder in Path, um Methoden wie .iterdir, .glob, .is_dir anzuwenden
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
        # CIMpy-Import -> in cim_data werden alle Informationen gespeichert, xml_files existiert nur kurz für jeden Schleifendurchlauf
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


