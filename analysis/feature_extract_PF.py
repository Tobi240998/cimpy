"""
PowerFactory Feature-Extractor

Erzeugt Features als Dictionary, konsistent mit CIM-Extractor.
Kompatibel mit compare_features.py.
"""

import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP7\Python\3.10")  # Pfad PowerFactory
import powerfactory as pf
import numpy as np

class PFFeatureExtractor:
    def __init__(self, project_name: str):
        # Verbindung zu PowerFactory herstellen
        self.app = pf.GetApplication()
        if self.app is None:
            raise RuntimeError("PowerFactory konnte nicht gefunden werden. Bitte Projekt öffnen!")

        # Projekt aktivieren
        self.app.ActivateProject(project_name)

        # Aktives Projekt abrufen
        self.project = self.app.GetActiveProject()
        if self.project is None:
            raise RuntimeError(f"Kein aktives Projekt gefunden: {project_name}")

        print(f"Aktives Projekt: {self.project.loc_name}")

        # Features-Dictionary vorbereiten
        self.features = {}

    def get_objects(self, obj_type: str):
        """Holt alle relevanten Objekte eines bestimmten Typs aus dem Projekt."""
        return self.app.GetCalcRelevantObjects(obj_type)

    def create_features(self) -> dict:
        """Erstellt den Feature-Vektor als Dictionary."""
        features = {}

        # -----------------------------
        # Objekte abrufen
        # -----------------------------
        buses = self.get_objects("*.ElmTerm")
        lines = self.get_objects("ElmLne")
        transformers = self.get_objects("ElmTr2")
        loads = self.get_objects("ElmLod")
        generators = self.get_objects("ElmGenstat")

        # Struktur-Features
        features['n_busbars'] = len(buses)
        features['n_lines'] = len(lines)
        features['n_transformers'] = len(transformers)
        features['n_loads'] = len(loads)
        features['n_generators'] = len(generators)

        # -----------------------------
        # Lastflussberechnung vorbereiten
        # -----------------------------
        ldf = self.app.GetFromStudyCase("ComLdf")
        ldf.Execute()

        # -----------------------------
        # Spannungen (Busbar-Spannungen)
        # -----------------------------
        voltages = [
            bus.GetAttribute("m:u")
            for bus in buses
            if bus.GetAttribute("m:u") is not None
        ]

        if voltages:
            v_array = np.array(voltages)
            features.update({
                "v_min": float(v_array.min()),
                "v_max": float(v_array.max()),
                "v_mean": float(v_array.mean()),
                "v_std": float(v_array.std()),
                "n_undervoltage": int((v_array < 0.95).sum()),
                "n_overvoltage": int((v_array > 1.05).sum()),
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
        # Lasten (Leistungsaufnahme)
        # -----------------------------
        P_loads = [
            load.GetAttribute("plini")
            for load in loads
            if load.GetAttribute("plini") is not None
        ]

        if P_loads:
            P_array = np.array(P_loads)
            features.update({
                "p_mean": float(P_array.mean()),
                "p_std": float(P_array.std()),
                "p_max_abs": float(np.abs(P_array).max()),
            })
        else:
            features.update({
                "p_mean": None,
                "p_std": None,
                "p_max_abs": None,
            })

        self.features = features
        return features

    def get_features(self) -> dict:
        """Gibt das Features-Dictionary zurück (berechnet es bei Bedarf)."""
        if not self.features:
            self.create_features()
        return self.features


