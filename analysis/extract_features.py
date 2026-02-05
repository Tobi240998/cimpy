"""
Extrahiert Feature-Vektoren aus CGMES/CIM-ZIP-Dateien
(kompatibel zu PowerFactory-Features & compare_features).
"""

import zipfile
import logging
from pathlib import Path
from collections import Counter
import cimpy


# -------------------------------------------------
# Logging (CIMpy ruhigstellen)
# -------------------------------------------------
logging.basicConfig(level=logging.ERROR)


# -------------------------------------------------
# ZIP entpacken (falls nötig)
# -------------------------------------------------
def unpack_zip(zip_path: Path) -> Path:
    target_dir = zip_path.with_suffix("")

    if not target_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print(f"ZIP entpackt nach: {target_dir}")
    else:
        print(f"ZIP bereits entpackt: {target_dir}")

    return target_dir


# -------------------------------------------------
# Struktur-Features
# -------------------------------------------------
def extract_structural_features(import_result: dict) -> dict:
    topology = import_result["topology"]

    class_counter = Counter(
        obj.__class__.__name__
        for obj in topology.values()
    )

    return {
        "n_busbars": class_counter.get("BusbarSection", 0),
        "n_lines": class_counter.get("ACLineSegment", 0),
        "n_transformers": class_counter.get("PowerTransformer", 0),
        "n_loads": (
            class_counter.get("EnergyConsumer", 0)
            + class_counter.get("ConformLoad", 0)
            + class_counter.get("LoadAggregate", 0)
            + class_counter.get("LoadStatic", 0)
        ),
        "n_generators": (
            class_counter.get("SynchronousMachine", 0)
            + class_counter.get("GeneratingUnit", 0)
        ),
    }


# -------------------------------------------------
# Snapshot-Features (SV)
# -------------------------------------------------
def extract_snapshot_features(import_result: dict) -> dict:
    topology = import_result["topology"]

    voltages = [
        obj.v
        for obj in topology.values()
        if obj.__class__.__name__ == "SvVoltage" and hasattr(obj, "v")
    ]

    flows_p = [
        obj.p
        for obj in topology.values()
        if obj.__class__.__name__ == "SvPowerFlow" and hasattr(obj, "p")
    ]

    features = {}

    # -----------------------------
    # Spannungen
    # -----------------------------
    if voltages:
        v_min = min(voltages)
        v_max = max(voltages)
        v_mean = sum(voltages) / len(voltages)
        variance = sum((v - v_mean) ** 2 for v in voltages) / len(voltages)
        v_std = variance ** 0.5

        features.update({
            "v_min": float(v_min),
            "v_max": float(v_max),
            "v_mean": float(v_mean),
            "v_std": float(v_std),
            "n_undervoltage": int(sum(v < 0.95 for v in voltages)),
            "n_overvoltage": int(sum(v > 1.05 for v in voltages)),
        })
    else:
        features.update({
            "v_min": 0.0,
            "v_max": 0.0,
            "v_mean": 0.0,
            "v_std": 0.0,
            "n_undervoltage": 0,
            "n_overvoltage": 0,
        })

    # -----------------------------
    # Leistungsflüsse
    # -----------------------------
    if flows_p:
        p_mean = sum(flows_p) / len(flows_p)
        variance = sum((p - p_mean) ** 2 for p in flows_p) / len(flows_p)
        p_std = variance ** 0.5
        p_max_abs = max(abs(p) for p in flows_p)

        features.update({
            "p_mean": float(p_mean),
            "p_std": float(p_std),
            "p_max_abs": float(p_max_abs),
        })
    else:
        features.update({
            "p_mean": 0.0,
            "p_std": 0.0,
            "p_max_abs": 0.0,
        })

    return features


# -------------------------------------------------
# Öffentliche API-Funktion (für compare_features)
# -------------------------------------------------
def extract_cim_features(zip_path) -> dict:
    """
    Extrahiert Feature-Vektor aus einem CIM-ZIP.
    Akzeptiert str oder Path.
    """
    if not isinstance(zip_path, Path):
        zip_path = Path(zip_path)

    work_dir = unpack_zip(zip_path)

    xml_files = sorted(str(p) for p in work_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError("Keine XML-Dateien gefunden")

    print(f"\nGefundene XML-Dateien: {len(xml_files)}")
    print("Starte CIMpy-Import ...")

    import_result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

    print("CIM erfolgreich importiert.")

    features = {}
    features.update(extract_structural_features(import_result))
    features.update(extract_snapshot_features(import_result))

    return features



# -------------------------------------------------
# Direkter Testlauf
# -------------------------------------------------
if __name__ == "__main__":
    zip_path = Path(
        r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\CIM_GridAssist_908.zip"
    )

    features = extract_cim_features(zip_path)

    print("\n--- Feature-Vektor (CIM) ---")
    for k, v in features.items():
        print(f"{k:<20}: {v}")
