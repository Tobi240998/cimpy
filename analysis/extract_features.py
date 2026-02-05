import zipfile
import logging
from pathlib import Path
from collections import Counter
import cimpy
from cimpy.cimimport import cim_import


# -------------------------------------------------------------------
# Logging (wie im CIMpy-Beispiel)
# -------------------------------------------------------------------
logging.basicConfig(
    filename="cim_import.log",
    level=logging.INFO,
    filemode="w"
)

# -------------------------------------------------------------------
# 1) Pfad zur ZIP-Datei
# -------------------------------------------------------------------
zip_path = Path(
    r"C:/Users/STELLER/Documents/Masterarbeit/CIM-Dateien/tobias_CIM_daten/data/CIM_GridAssist_908.zip"
)

assert zip_path.exists(), "ZIP-Datei nicht gefunden!"

# -------------------------------------------------------------------
# 2) ZIP entpacken
# -------------------------------------------------------------------
extract_dir = zip_path.parent / zip_path.stem
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)

print(f"ZIP entpackt nach: {extract_dir}")

# -------------------------------------------------------------------
# 3) XML-Dateien sammeln
# -------------------------------------------------------------------
xml_files = [str(p.resolve()) for p in extract_dir.glob("*.xml")]

print(f"\nGefundene XML-Dateien: {len(xml_files)}")
for f in xml_files:
    print(" -", Path(f).name)

if not xml_files:
    raise RuntimeError("Keine XML-Dateien gefunden – Abbruch")

# -------------------------------------------------------------------
# 4) CIMpy Import (wie im offiziellen Beispiel)
# -------------------------------------------------------------------
print("\nStarte CIMpy-Import ...")

import_result = cim_import(
    xml_files,
    cgmes_version="cgmes_v2_4_15"
)

print("CIM erfolgreich importiert.")
print("Keys im Import-Result:", import_result.keys())

# -------------------------------------------------------------------
# 5) Inhalt der Topologie analysieren
# -------------------------------------------------------------------
topology = import_result.get("topology", {})

print(f"\nAnzahl Objekte in topology: {len(topology)}")

class_counter = Counter()
for obj in topology.values():
    class_counter[obj.__class__.__name__] += 1

print("\nGefundene CIM-Klassen:")
for cls, count in class_counter.most_common():
    print(f"{cls:30s} {count}")

# -------------------------------------------------------------------
# 6) Gezielt prüfen: relevante Klassen
# -------------------------------------------------------------------
interesting = [
    "BusbarSection",
    "ACLineSegment",
    "PowerTransformer",
    "EnergyConsumer",
    "Terminal",
]

print("\nRelevante Klassen:")
for cls in interesting:
    print(f"{cls:30s} {class_counter.get(cls, 0)}")


def extract_structural_features(import_result):
    topology = import_result["topology"]

    class_names = [obj.__class__.__name__ for obj in topology.values()]
    counts = Counter(class_names)

    n_buses = counts.get("TopologicalNode", 0)
    n_lines = counts.get("ACLineSegment", 0)
    n_transformers = counts.get("PowerTransformer", 0)
    n_generators = (
        counts.get("SynchronousMachine", 0)
        + counts.get("GeneratingUnit", 0)
    )
    n_loads = counts.get("EnergyConsumer", 0) + counts.get("ConformLoad", 0)

    features = {
        "n_buses": n_buses,
        "n_lines": n_lines,
        "n_transformers": n_transformers,
        "n_generators": n_generators,
        "n_loads": n_loads,
        "lines_per_bus": n_lines / n_buses if n_buses else 0,
        "transformers_per_bus": n_transformers / n_buses if n_buses else 0,
        "generators_per_bus": n_generators / n_buses if n_buses else 0,
    }

    return features
features = extract_structural_features(import_result)
print("Feature-Vektor:")
for k, v in features.items():
    print(f"{k:25s}: {v}")
