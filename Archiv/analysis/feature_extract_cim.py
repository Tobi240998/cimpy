import logging  # steuert die Ausgaben, wird benötigt um nicht hunderte von Fehlermeldungen vom CIM-Import zu erhalten
from pathlib import Path  # plattformunabhängige Nutzung von Pfaden möglich 
from collections import Counter  # zum Zählen von Objekten 
import cimpy  # zur Erstellung von Python-Objekten aus XML

logging.basicConfig(level=logging.ERROR)  # nur echte Fehler werden angezeigt, Warnungen ausgeblendet 

class CIMFeatureExtractor:
    def __init__(self, extracted_folder: str):
        
        self.extracted_folder = Path(extracted_folder)

    def extract_features(self) -> dict:
        
        # Extrahiert die Features aus allen entpackten CIM-Daten. Pro Netz (Ordner) wird ein Feature-Dictionary erzeugt.
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

            features = {
                "structure": {},
                "state": {}
            }

            # -----------------------------
            # Struktur -> Zählen der jeweiligen Anzahl der Objekte 
            # -----------------------------
            counter = Counter(obj.__class__.__name__ for obj in topology.values())

            features["structure"].update({
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
            # Installierte Last
            # -----------------------------
            loads_p_inst = [
                obj.p for obj in topology.values()
                if obj.__class__.__name__ == "EnergyConsumer" and hasattr(obj, "p")
            ]

            features["structure"].update({
                "P_load_installed": sum(loads_p_inst) if loads_p_inst else 0.0
            })

            # -----------------------------
            # Verhältnisse Struktur
            # -----------------------------
            n_buses = features["structure"]["n_busbars"] or 1  # Schutz gegen Division durch 0
            n_lines = features["structure"]["n_lines"]
            n_transformers = features["structure"]["n_transformers"]
            n_loads = features["structure"]["n_loads"]
            n_generators = features["structure"]["n_generators"]
            P_load_installed = features["structure"]["P_load_installed"]

            features["structure"].update({
                "lines_per_bus": n_lines / n_buses,
                "transformers_per_bus": n_transformers / n_buses,
                "loads_per_bus": n_loads / n_buses,
                "gens_per_bus": n_generators / n_buses,
                "P_load_per_bus": P_load_installed / n_buses
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

                features["state"].update({
                    "v_min": min(voltages),
                    "v_max": max(voltages),
                    "v_mean": v_mean,
                    "v_std": v_std,
                    "n_undervoltage": sum(v < 0.95 for v in voltages),
                    "n_overvoltage": sum(v > 1.05 for v in voltages),
                })

                # Verhältnis-Features
                n_buses_nonzero = len(voltages) or 1
                features["state"].update({
                    "share_undervoltage": sum(v < 0.95 for v in voltages) / n_buses_nonzero,
                    "share_overvoltage": sum(v > 1.05 for v in voltages) / n_buses_nonzero,
                    "v_range": max(voltages) - min(voltages)
                })
            else:
                features["state"].update({
                    "v_min": None,
                    "v_max": None,
                    "v_mean": None,
                    "v_std": None,
                    "n_undervoltage": 0,
                    "n_overvoltage": 0,
                    "share_undervoltage": 0,
                    "share_overvoltage": 0,
                    "v_range": 0
                })

            if flows_p:
                p_array = [abs(p) for p in flows_p]
                features["state"].update({
                    "p_mean_abs": sum(p_array) / len(p_array),
                    "p_max_abs": max(p_array)
                })
            else:
                features["state"].update({
                    "p_mean_abs": None,
                    "p_max_abs": None
                })

            # Features dieses CIM-Falls speichern
            all_features[case_dir.name] = features

        return all_features
