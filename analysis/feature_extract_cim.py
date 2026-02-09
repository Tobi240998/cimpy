import zipfile
import logging
from pathlib import Path
from collections import Counter
import cimpy

logging.basicConfig(level=logging.ERROR)


class CIMFeatureExtractor:
    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)

    def unpack_zip(self) -> Path:
        target_dir = self.zip_path.with_suffix("")
        if not target_dir.exists():
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                zf.extractall(target_dir)
        return target_dir

    def extract_features(self) -> dict:
        work_dir = self.unpack_zip()
        xml_files = sorted(str(p) for p in work_dir.glob("*.xml"))
        if not xml_files:
            raise RuntimeError("Keine XML-Dateien gefunden")

        import_result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")
        topology = import_result["topology"]

        features = {}

        # -----------------------------
        # Struktur
        # -----------------------------
        counter = Counter(obj.__class__.__name__ for obj in topology.values())

        features.update({
            "n_busbars": counter.get("BusbarSection", 0),
            "n_lines": counter.get("ACLineSegment", 0),
            "n_transformers": counter.get("PowerTransformer", 0),
            "n_loads": (
                counter.get("EnergyConsumer", 0)
                + counter.get("ConformLoad", 0)
                + counter.get("LoadStatic", 0)
            ),
            "n_generators": (
                counter.get("SynchronousMachine", 0)
                + counter.get("GeneratingUnit", 0)
            ),
        })

        # -----------------------------
        # Snapshots
        # -----------------------------
        voltages = [
            obj.v for obj in topology.values()
            if obj.__class__.__name__ == "SvVoltage" and hasattr(obj, "v")
        ]

        flows_p = [
            obj.p for obj in topology.values()
            if obj.__class__.__name__ == "SvPowerFlow" and hasattr(obj, "p")
        ]

        if voltages:
            v_mean = sum(voltages) / len(voltages)
            v_std = (sum((v - v_mean) ** 2 for v in voltages) / len(voltages)) ** 0.5

            features.update({
                "v_min": min(voltages),
                "v_max": max(voltages),
                "v_mean": v_mean,
                "v_std": v_std,
                "n_undervoltage": sum(v < 0.95 for v in voltages),
                "n_overvoltage": sum(v > 1.05 for v in voltages),
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

        if flows_p:
            p_mean = sum(flows_p) / len(flows_p)
            p_std = (sum((p - p_mean) ** 2 for p in flows_p) / len(flows_p)) ** 0.5

            features.update({
                "p_mean": p_mean,
                "p_std": p_std,
                "p_max_abs": max(abs(p) for p in flows_p),
            })
        else:
            features.update({
                "p_mean": None,
                "p_std": None,
                "p_max_abs": None,
            })

        return features
