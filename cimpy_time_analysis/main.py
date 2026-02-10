import os
import cimpy
from llm_cim_orchestrator import handle_user_query
from cim_snapshot_cache import preprocess_snapshots
from load_cim_data import load_cim_snapshots


if __name__ == "__main__":
    cim_root = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    cim_snapshots = load_cim_snapshots(cim_root)

    snapshot_cache = preprocess_snapshots(cim_snapshots)


    user_input = "Wie verhält sich die Trafo-Leistung über den Tag?"

    answer = handle_user_query(user_input, snapshot_cache)

    print("\nAntwort:\n")
    print(answer)
