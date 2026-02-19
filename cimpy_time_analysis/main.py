from llm_cim_orchestrator import handle_user_query
from cim_snapshot_cache import preprocess_snapshots
from load_cim_data import load_cim_snapshots
from load_cim_data import build_network_index


if __name__ == "__main__":
    cim_root = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    # CIM-Daten laden
    cim_snapshots = load_cim_snapshots(cim_root)

    # Daten einmalig vorverarbeiten
    snapshot_cache = preprocess_snapshots(cim_snapshots)
    network_index = build_network_index(cim_snapshots)


   

    # Nutzerfrage -> später Umstellung auf LLM
    user_input = "Wie verhält sich die Leistung von Transformator 19-20 über den Tag?"

    # Orchestrator aufrufen 
    answer = handle_user_query(
        user_input,
        snapshot_cache,
        network_index
    )

    # Ausgabe
    print("\nAntwort:\n")
    print(answer)
