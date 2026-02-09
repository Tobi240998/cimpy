import logging  # steuert die Ausgaben, wird benötigt um nicht hunderte von Fehlermeldungen vom CIM-Import zu erhalten
from pathlib import Path  # plattformunabhängige Nutzung von Pfaden möglich 
from collections import Counter  # zum Zählen von Objekten 
import cimpy  # zur Erstellung von Python-Objekten aus XML


logging.basicConfig(level=logging.ERROR)  # nur echte Fehler werden angezeigt, Warnungen ausgeblendet 


class CIMFeatureExtractor:
    def __init__(self, extracted_folder: str):
        """
        Initialisiert den CIMFeatureExtractor.
        extracted_folder: Pfad zum Ordner mit allen entpackten CIM-Fällen
        """
        self.extracted_folder = Path(extracted_folder)

    def extract_features(self) -> dict:
        """
        Extrahiert die Features aus allen entpackten CIM-Fällen.
        Pro Netz (Ordner) wird ein Feature-Dictionary erzeugt.
        """
        all_features = {}

        # Alle entpackten Netz-Ordner sammeln
        case_dirs = sorted(p for p in self.extracted_folder.iterdir() if p.is_dir())
        print(f"Gefundene entpackte CIM-Fälle: {len(case_dirs)}")

        for case_dir in case_dirs:
            # ---------------------------------------------
            # XML-Dateien für diesen CIM-Fall sammeln
            # ---------------------------------------------
            xml_files = sorted(str(p) for p in case_dir.glob("*.xml"))
            if not xml_files:
                print(f"Keine XML-Dateien in {case_dir.name}, überspringe...")
                continue

            # ---------------------------------------------
            # CIMpy-Import
            # ---------------------------------------------
            import_result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")
            topology = import_result["topology"]

            features = {}

            # -----------------------------
            # Struktur -> Zählen der jeweiligen Anzahl der Objekte 
            # -----------------------------
            counter = Counter(obj.__class__.__name__ for obj in topology.values())

            features.update({
                "n_busbars": counter.get("BusbarSection", 0),
                "n_lines": counter.get("ACLineSegment", 0),
                "n_transformers": counter.get("PowerTransformer", 0),
                "n_loads": (
                    counter.get("EnergyConsumer", 0)
                    + counter.get("ConformLoad", 0)
                    + counter.get("LoadStatic", 0)  # Berücksichtigung der verschiedenen Last-Typen von CIM
                ),
                "n_generators": (
                    counter.get("SynchronousMachine", 0)
                    + counter.get("GeneratingUnit", 0)
                ),
            })

            # -----------------------------
            # Snapshots
            # -----------------------------
            # Sammlung der Spannungen
            voltages = [
                obj.v for obj in topology.values()
                if obj.__class__.__name__ == "SvVoltage" and hasattr(obj, "v")
            ]

            # Sammlung der Leistungen
            flows_p = [
                obj.p for obj in topology.values()
                if obj.__class__.__name__ == "SvPowerFlow" and hasattr(obj, "p")
            ]

            # Extraktion der Features aus den gesammelten Spannungen / Lasten 
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

            # Features dieses CIM-Falls speichern
            all_features[case_dir.name] = features

        return all_features
