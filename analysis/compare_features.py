import numpy as np
from feature_extract_pf import PFFeatureExtractor
from feature_extract_cim import CIMFeatureExtractor


class FeatureComparer:
    def __init__(self, cim_zip: str, pf_project: str):
        self.cim_zip = cim_zip
        self.pf_project = pf_project

        self.features_cim = None
        self.features_pf = None

    def extract_features(self):
        self.features_cim = CIMFeatureExtractor(self.cim_zip).extract_features()
        self.features_pf = PFFeatureExtractor(self.pf_project).create_features()

    def _to_vector(self, feature_dict, keys):
        return np.array([
            feature_dict[k] if feature_dict[k] is not None else 0.0
            for k in keys
        ], dtype=float)

    def compare(self):
        keys = sorted(set(self.features_cim) & set(self.features_pf))

        v_cim = self._to_vector(self.features_cim, keys)
        v_pf = self._to_vector(self.features_pf, keys)

        diff = v_pf - v_cim
        dist = np.linalg.norm(diff)

        print("\n--- Feature-Vergleich ---")
        for k, a, b, d in zip(keys, v_cim, v_pf, diff):
            print(f"{k:20s} CIM={a:8.3f} PF={b:8.3f} Δ={d:8.3f}")

        print("\nEuklidische Distanz:", dist)


# -------------------------------------------------
# Start
# -------------------------------------------------
if __name__ == "__main__":
    comparer = FeatureComparer(
        cim_zip=r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\CIM_GridAssist_908.zip",
        pf_project="Nine-bus System(2)"
    )

    comparer.extract_features()
    comparer.compare()
