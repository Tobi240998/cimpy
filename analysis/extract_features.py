import zipfile
import logging
from pathlib import Path
from collections import Counter
import cimpy


# -------------------------------------------------
# Logging (unterdrückt CIMpy-Spam, aber behält Fehler)
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

    features = {
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

    return features


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
            "v_min": v_min,
            "v_max": v_max,
            "v_mean": v_mean,
            "v_std": v_std,
            "n_undervoltage": sum(1 for v in voltages if v < 0.95),
            "n_overvoltage": sum(1 for v in voltages if v > 1.05),
        })
    else:
        features.update({
            "v_min": None,
            "v_max": None,
            "v_mean": None,
            "v_std": None,
            "n_undervoltage": 0,
            "n_overvoltage": 0,
        })

    # -----------------------------
    # Leistungsflüsse
    # -----------------------------
    if flows_p:
        p_mean = sum(flows_p) / len(flows_p)
        p_max_abs = max(abs(p) for p in flows_p)

        variance = sum((p - p_mean) ** 2 for p in flows_p) / len(flows_p)
        p_std = variance ** 0.5

        features.update({
            "p_mean": p_mean,
            "p_std": p_std,
            "p_max_abs": p_max_abs,
        })
    else:
        features.update({
            "p_mean": None,
            "p_std": None,
            "p_max_abs": None,
        })

    return features



# -------------------------------------------------
# Hauptfunktion: Feature-Extraktion
# -------------------------------------------------
def extract_features_from_zip(zip_path: Path) -> dict:
    work_dir = unpack_zip(zip_path)

    xml_files = sorted(str(p) for p in work_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError("Keine XML-Dateien gefunden")

    print(f"\nGefundene XML-Dateien: {len(xml_files)}")

    print("\nStarte CIMpy-Import ...")
    import_result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

    print("CIM erfolgreich importiert.")

    features = {}
    features.update(extract_structural_features(import_result))
    features.update(extract_snapshot_features(import_result))

    return features


# -------------------------------------------------
# Testlauf
# -------------------------------------------------
if __name__ == "__main__":
    zip_path = Path(
        r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\CIM_GridAssist_908.zip"
    )

    features = extract_features_from_zip(zip_path)

    print("\n--- Feature-Vektor ---")
    for k, v in features.items():
        print(f"{k:20s}: {v}")
