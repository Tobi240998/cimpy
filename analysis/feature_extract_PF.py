"""
Extrahiert Features direkt aus dem aktiven PowerFactory-Projekt
und gibt sie als Dictionary zurück (für Vergleich mit CIM).
"""

import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP7\Python\3.10")

import powerfactory as pf
import numpy as np


def extract_pf_features(project_name=None):
    """
    Extrahiert Feature-Vektor aus aktivem PowerFactory-Projekt.
    Falls project_name gesetzt ist, wird dieses Projekt aktiviert.
    """

    # -------------------------------------------------
    # Verbindung zu PowerFactory
    # -------------------------------------------------
    app = pf.GetApplication()
    if app is None:
        raise RuntimeError("PowerFactory konnte nicht gefunden werden.")

    #Projekt aktivieren 
    project_name = "Nine-bus System(2)"
    app.ActivateProject(project_name)

    project = app.GetActiveProject()
    if project is None:
        raise RuntimeError("Kein aktives Projekt gefunden.")

    print(f"Aktives Projekt: {project.loc_name}")

    # -------------------------------------------------
    # Hilfsfunktion
    # -------------------------------------------------
    def get_objects(pattern):
        objs = app.GetCalcRelevantObjects(pattern)
        return objs if objs is not None else []

    # -------------------------------------------------
    # Lastfluss rechnen (wichtig für m:* Attribute)
    # -------------------------------------------------
    ldf = app.GetFromStudyCase("ComLdf")
    if ldf is None:
        raise RuntimeError("Kein Load Flow (ComLdf) im Study Case gefunden.")
    ldf.Execute()

    # -------------------------------------------------
    # Feature-Dictionary
    # -------------------------------------------------
    features = {}

    # -------------------------------------------------
    # Struktur-Features
    # -------------------------------------------------
    buses = get_objects("*.ElmTerm")
    lines = get_objects("*.ElmLne")
    transformers = get_objects("*.ElmTr2")
    loads = get_objects("*.ElmLod")
    generators = get_objects("*.ElmGenstat")

    features["n_busbars"] = len(buses)
    features["n_lines"] = len(lines)
    features["n_transformers"] = len(transformers)
    features["n_loads"] = len(loads)
    features["n_generators"] = len(generators)

    # -------------------------------------------------
    # Spannungs-Features (Snapshot)
    # -------------------------------------------------
    voltages = np.array([
        bus.GetAttribute("m:u")
        for bus in buses
        if bus.GetAttribute("m:u") is not None
    ])

    if len(voltages) > 0:
        features["v_min"] = float(voltages.min())
        features["v_max"] = float(voltages.max())
        features["v_mean"] = float(voltages.mean())
        features["v_std"] = float(voltages.std())
        features["n_undervoltage"] = int((voltages < 0.9 * voltages.mean()).sum())
        features["n_overvoltage"] = int((voltages > 1.1 * voltages.mean()).sum())
    else:
        features["v_min"] = 0.0
        features["v_max"] = 0.0
        features["v_mean"] = 0.0
        features["v_std"] = 0.0
        features["n_undervoltage"] = 0
        features["n_overvoltage"] = 0

    # -------------------------------------------------
    # Last-Features
    # -------------------------------------------------
    P_loads = np.array([
        load.GetAttribute("plini")
        for load in loads
        if load.GetAttribute("plini") is not None
    ])

    if len(P_loads) > 0:
        features["p_mean"] = float(P_loads.mean())
        features["p_std"] = float(P_loads.std())
        features["p_max_abs"] = float(np.abs(P_loads).max())
    else:
        features["p_mean"] = 0.0
        features["p_std"] = 0.0
        features["p_max_abs"] = 0.0

    return features


# -------------------------------------------------
# Direkter Testlauf
# -------------------------------------------------
if __name__ == "__main__":
    features = extract_pf_features(project_name="Nine-bus System(2)")

    print("\n--- Feature-Vektor aus PowerFactory ---")
    for k, v in features.items():
        print(f"{k:<20}: {v}")
