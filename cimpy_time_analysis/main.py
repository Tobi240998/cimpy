# main.py

import os
import cimpy
from llm_cim_orchestrator import handle_user_query


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

        print(f"Lade Snapshot {case} ({len(xml_files)} XML-Dateien)")
        snapshots[case] = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

    return snapshots


if __name__ == "__main__":
    cim_root = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    cim_snapshots = load_cim_snapshots(cim_root)

    user_input = "Wie verhält sich die Trafo-Leistung über den Tag?"

    answer = handle_user_query(user_input, cim_snapshots)
    print("\nAntwort:\n")
    print(answer)
