# cim_historical/runner.py

from cimpy.cimpy_time_analysis.llm_cim_orchestrator import handle_user_query
from cimpy.cimpy_time_analysis.cim_snapshot_cache import preprocess_snapshots
from cimpy.cimpy_time_analysis.load_cim_data import load_cim_snapshots, build_network_index


def run_historical_cim_analysis(
    user_input: str,
    cim_root: str,
    preloaded_snapshots=None,
):
    """
    Führt eine historische CIM-Analyse aus.

    Parameters
    ----------
    user_input : str
        Nutzerfrage zur Analyse.
    cim_root : str
        Root-Pfad zu den extrahierten CIM-Dateien.
    preloaded_snapshots : optional
        Bereits geladene Snapshots (für Performance / Router-Betrieb).

    Returns
    -------
    dict
        Standardisiertes Result-Objekt.
    """

    # 1. Snapshots laden (falls nicht bereits vorhanden)
    if preloaded_snapshots is None:
        cim_snapshots = load_cim_snapshots(cim_root)
    else:
        cim_snapshots = preloaded_snapshots

    # 2. Preprocessing
    snapshot_cache = preprocess_snapshots(cim_snapshots)
    network_index = build_network_index(cim_snapshots)

    # 3. Analyse
    answer = handle_user_query(
        user_input,
        snapshot_cache,
        network_index
    )

    return {
        "status": "ok",
        "tool": "historical_cim_analysis",
        "input": user_input,
        "answer": answer,
    }