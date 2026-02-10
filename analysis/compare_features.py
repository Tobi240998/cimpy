import numpy as np #für Rechnungen bei Vektorvergleich
from feature_extract_pf import PFFeatureExtractor #Import der Klasse zum Feature Extract von PowerFactory
from feature_extract_cim import CIMFeatureExtractor #Import der Klasse zum Feature Extract von CIM


class FeatureComparer:
    def __init__(self, cim_folder: str, pf_project: str):
        self.cim_folder = cim_folder
        self.pf_project = pf_project

        self.features_cim_all = None
        self.features_pf = None

    #Zugriff auf die Funktionen, die die Features erstellen
    def extract_features(self):
        self.features_cim_all = CIMFeatureExtractor(self.cim_folder).extract_features()
        self.features_pf = PFFeatureExtractor(self.pf_project).create_features()
    
    #Umwandlung von dict in Vektor
    def _to_vector(self, feature_dict, keys):
        return np.array([
            feature_dict[k] if feature_dict[k] is not None else 0.0
            for k in keys
        ], dtype=float)

    def _z_score_normalize(self, matrix):
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        std[std == 0] = 1.0  #Schutz gegen Division durch 0
        return (matrix - mean) / std

    def compare(self):
        for category in ["structure", "state"]:
            print(f"\n--- Vergleich Kategorie: {category.upper()} ---")

            keys = sorted(
                set(self.features_pf[category].keys()) &
                set(next(iter(self.features_cim_all.values()))[category].keys())
            )

            cim_names = []
            cim_vectors = []

            for name, features_cim in self.features_cim_all.items():
                cim_names.append(name)
                cim_vectors.append(self._to_vector(features_cim[category], keys))

            cim_vectors = np.array(cim_vectors)
            v_pf = self._to_vector(self.features_pf[category], keys)

            all_vectors = np.vstack([cim_vectors, v_pf])
            all_vectors_norm = self._z_score_normalize(all_vectors)

            cim_vectors_norm = all_vectors_norm[:-1]
            v_pf_norm = all_vectors_norm[-1]

            best_name = None
            best_dist = None

            for name, v_cim in zip(cim_names, cim_vectors_norm):
                diff = v_pf_norm - v_cim
                dist = np.linalg.norm(diff)

                print(f"{name:30s} Distanz = {dist:.6f}")

                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_name = name

            print(f"\nBestes Matching ({category}): {best_name}")
            print(f"Euklidische Distanz: {best_dist:.6f}")
