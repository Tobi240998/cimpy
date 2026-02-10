"""
PowerFactory Feature-Extractor

Erzeugt Features als Dictionary, konsistent mit CIM-Extractor.

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
        # Holt alle kalkulationsrelevanten Objekte eines bestimmten Typs aus dem Projekt - für State-Vergleich 
        return self.app.GetCalcRelevantObjects(obj_type)

    def get_all_objects(self, obj_type: str):
        # Holt alle Objekte eines bestimmten Typs aus dem Projekt - für Struktur-Vergleich
        return self.project.GetContents(obj_type, recursive=True)

    def create_features(self) -> dict:
        """Erstellt den Feature-Vektor als Dictionary."""
        features = {
            "structure": {},
            "state": {}
        }

        # -----------------------------
        # Objekte abrufen
        # -----------------------------
        # Struktur: alle Objekte
        buses_all = self.get_all_objects("*.ElmTerm")
        lines_all = self.get_all_objects("*.ElmLne")
        transformers_all = self.get_all_objects("*.ElmTr2")
        loads_all = self.get_all_objects("*.ElmLod")
        generators_all = self.get_all_objects("*.ElmGenstat")

        # State: nur calcrelevant Objekte
        buses = self.get_objects("*.ElmTerm")
        lines = self.get_objects("ElmLne")

        # Struktur-Features
        features['structure'].update({
            'n_busbars': len(buses_all),
            'n_lines': len(lines_all),
            'n_transformers': len(transformers_all),
            'n_loads': len(loads_all),
            'n_generators': len(generators_all)
        })

       
        # Installierte Last
        P_load_installed = sum(
            load.GetAttribute("plini")
            for load in loads_all
            if load.GetAttribute("plini") is not None
        )

        features["structure"].update({
            "P_load_installed": float(P_load_installed)
        })

        
        # Verhältnisse Struktur
        n_buses = len(buses_all) or 1  # Schutz gegen Division durch 0
        n_lines = len(lines_all)
        n_transformers = len(transformers_all)
        n_loads = len(loads_all)
        n_generators = len(generators_all)

        features["structure"].update({
            "lines_per_bus": n_lines / n_buses,
            "transformers_per_bus": n_transformers / n_buses,
            "loads_per_bus": n_loads / n_buses,
            "gens_per_bus": n_generators / n_buses,
            "P_load_per_bus": P_load_installed / n_buses
        })


        # Lastflussberechnung 
        ldf = self.app.GetFromStudyCase("ComLdf")
        ldf.Execute()

        # Spannungen (Busbar-Spannungen)
        voltages = [
            bus.GetAttribute("m:u")
            for bus in buses
            if bus.GetAttribute("m:u") is not None
        ]

        if voltages:
            v_array = np.array(voltages)
            features["state"].update({
                "v_min": float(v_array.min()),
                "v_max": float(v_array.max()),
                "v_mean": float(v_array.mean()),
                "v_std": float(v_array.std()),
                "n_undervoltage": int((v_array < 0.95).sum()),
                "n_overvoltage": int((v_array > 1.05).sum()),
                "share_undervoltage": float((v_array < 0.95).sum()) / len(v_array),
                "share_overvoltage": float((v_array > 1.05).sum()) / len(v_array),
                "v_range": float(v_array.max() - v_array.min())
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

        # -----------------------------
        # Leistungsflüsse (Snapshot)
        # -----------------------------
        flows = []
        for line in lines:
            p1 = line.GetAttribute("m:P:bus1")
            p2 = line.GetAttribute("m:P:bus2")
            if p1 is not None:
                flows.append(abs(p1))
            if p2 is not None:
                flows.append(abs(p2))

        if flows:
            flows_array = np.array(flows)
            features["state"].update({
                "p_mean_abs": float(flows_array.mean()),
                "p_max_abs": float(flows_array.max())
            })
        else:
            features["state"].update({
                "p_mean_abs": None,
                "p_max_abs": None
            })

        self.features = features
        return features


