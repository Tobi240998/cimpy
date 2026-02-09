import sys
import numpy as np

# PowerFactory Python-Pfad
sys.path.append(
    r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP7\Python\3.10"
)
import powerfactory as pf


class PFFeatureExtractor:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.app = pf.GetApplication()
        if self.app is None:
            raise RuntimeError("PowerFactory konnte nicht gefunden werden.")

        self.app.ActivateProject(project_name)
        self.project = self.app.GetActiveProject()
        if self.project is None:
            raise RuntimeError("Kein aktives Projekt.")

    def get_objects(self, obj_type: str):
        return self.app.GetCalcRelevantObjects(obj_type)

    def create_features(self) -> dict:
        features = {}

        # -----------------------------
        # Struktur
        # -----------------------------
        buses = self.get_objects("*.ElmTerm")
        lines = self.get_objects("*.ElmLne")
        transformers = self.get_objects("*.ElmTr2")
        loads = self.get_objects("*.ElmLod")
        generators = self.get_objects("*.ElmGenstat")

        features.update({
            "n_busbars": len(buses),
            "n_lines": len(lines),
            "n_transformers": len(transformers),
            "n_loads": len(loads),
            "n_generators": len(generators),
        })

        # -----------------------------
        # Lastflussrechnung
        # -----------------------------
        ldf = self.app.GetFromStudyCase("ComLdf")
        ldf.Execute()

        # -----------------------------
        # Spannungen
        # -----------------------------
        voltages = np.array([
            bus.GetAttribute("m:u")
            for bus in buses
            if bus.GetAttribute("m:u") is not None
        ])

        if voltages.size > 0:
            features.update({
                "v_min": voltages.min(),
                "v_max": voltages.max(),
                "v_mean": voltages.mean(),
                "v_std": voltages.std(),
                "n_undervoltage": (voltages < 0.95).sum(),
                "n_overvoltage": (voltages > 1.05).sum(),
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
        # Lasten
        # -----------------------------
        P_loads = np.array([
            load.GetAttribute("plini")
            for load in loads
            if load.GetAttribute("plini") is not None
        ])

        if P_loads.size > 0:
            features.update({
                "p_mean": P_loads.mean(),
                "p_std": P_loads.std(),
                "p_max_abs": np.abs(P_loads).max(),
            })
        else:
            features.update({
                "p_mean": None,
                "p_std": None,
                "p_max_abs": None,
            })

        return features
