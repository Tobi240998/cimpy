"""


Extrahiert Features direkt aus einem aktiven PowerFactory-Projekt.
"""

import powerfactory as pf
import numpy as np

# Verbindung zu PowerFactory herstellen
app = pf.GetApplication()
if app is None:
    raise RuntimeError("PowerFactory konnte nicht gefunden werden. Bitte Projekt öffnen!")

# Aktives Projekt holen
project = app.GetActiveProject()
if project is None:
    raise RuntimeError("Kein aktives Projekt gefunden. Bitte Projekt öffnen!")

print(f"Aktives Projekt: {project.loc_name}")

# Funktion, um Objekte aus PowerFactory zu holen
def get_objects(obj_type):
    return app.GetCalcRelevantObjects(obj_type)

features = {}

# Struktur-Features
buses = get_objects("ElmBus")
features['n_busbars'] = len(buses)

lines = get_objects("ElmLne")
features['n_lines'] = len(lines)

transformers = get_objects("ElmTr2")
features['n_transformers'] = len(transformers)

loads = get_objects("ElmLod")
features['n_loads'] = len(loads)

generators = get_objects("ElmGenstat")
features['n_generators'] = len(generators)

# Spannungs-Features
voltages = np.array([bus.GetAttribute('m:u') for bus in buses if bus.GetAttribute('m:u') is not None])
if len(voltages) > 0:
    features['v_min'] = voltages.min()
    features['v_max'] = voltages.max()
    features['v_mean'] = voltages.mean()
    features['v_std'] = voltages.std()
    features['n_undervoltage'] = (voltages < 0.9 * voltages.mean()).sum()
    features['n_overvoltage'] = (voltages > 1.1 * voltages.mean()).sum()
else:
    features['v_min'] = features['v_max'] = features['v_mean'] = features['v_std'] = 0
    features['n_undervoltage'] = features['n_overvoltage'] = 0

# Last-Features
P_loads = np.array([load.GetAttribute('m:P') for load in loads if load.GetAttribute('m:P') is not None])
if len(P_loads) > 0:
    features['p_mean'] = P_loads.mean()
    features['p_std'] = P_loads.std()
    features['p_max_abs'] = np.abs(P_loads).max()
else:
    features['p_mean'] = features['p_std'] = features['p_max_abs'] = 0

# Ausgabe
print("\n--- Feature-Vektor aus aktivem PowerFactory-Projekt ---")
for k, v in features.items():
    print(f"{k:<20}: {v}")
