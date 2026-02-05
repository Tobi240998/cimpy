import math
from extract_features import extract_cim_features
from feature_extract_PF import extract_pf_features

# ---------------------------------------------------------
# Feature-Reihenfolge (einheitlich!)
# ---------------------------------------------------------

FEATURE_ORDER = [
    "n_busbars",
    "n_lines",
    "n_transformers",
    "n_loads",
    "n_generators",
    "v_min",
    "v_max",
    "v_mean",
    "v_std",
    "n_undervoltage",
    "n_overvoltage",
    "p_mean",
    "p_std",
    "p_max_abs",
]

# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def dict_to_vector(feature_dict):
    vector = []
    for key in FEATURE_ORDER:
        if key not in feature_dict:
            raise KeyError(f"Feature fehlt: {key}")
        vector.append(float(feature_dict[key]))
    return vector


def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# ---------------------------------------------------------
# HAUPTPROGRAMM
# ---------------------------------------------------------

if __name__ == "__main__":

    # --- 1) CIM-Features ---
    cim_zip = r"C:/Users/STELLER/Documents/Masterarbeit/CIM-Dateien/tobias_CIM_daten/data/CIM_GridAssist_908.zip"
    features_cim = extract_cim_features(cim_zip)

    # --- 2) PowerFactory-Features ---
    features_pf = extract_pf_features()

    # --- 3) Vektoren bauen ---
    vec_cim = dict_to_vector(features_cim)
    vec_pf = dict_to_vector(features_pf)

    # --- 4) Vergleich ---
    distance = euclidean_distance(vec_pf, vec_cim)

    # --- 5) Ergebnis ---
    print("\n--- Vergleich PF ↔ CIM ---")
    print(f"Euklidische Distanz: {distance:.3f}")

    if distance < 10:
        print("→ Sehr ähnliche Netze / Betriebssituationen")
    elif distance < 50:
        print("→ Teilweise ähnlich")
    else:
        print("→ Deutlich unterschiedliche Netze")
