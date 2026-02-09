import numpy as np #für Rechnungen bei Vektorvergleich
from feature_extract_pf import PFFeatureExtractor #Import der Klasse zum Feature Extract von PowerFactory
from feature_extract_cim import CIMFeatureExtractor #Import der Klasse zum Feature Extract von CIM
from pathlib import Path #plattformunabhängige Nutzung von Pfaden möglich


class FeatureComparer:
    def __init__(self, cim_folder: str, pf_project: str):
        """
        Initialisiert den FeatureComparer.
        cim_folder: Pfad zum Ordner mit allen CIM-ZIP-Dateien
        pf_project: Name des aktiven PowerFactory-Projekts
        """
        self.cim_folder = Path(cim_folder)
        self.pf_project = pf_project

        self.features_cim_all = None
        self.features_pf = None

    #Zugriff auf die Funktionen, die die Features erstellen
    def extract_features(self):
        #Alle CIM-ZIP-Dateien im Ordner werden extrahiert und Features erstellt
        self.features_cim_all = CIMFeatureExtractor(self.cim_folder).extract_features()
        #PowerFactory-Features
        self.features_pf = PFFeatureExtractor(self.pf_project).create_features()

    def _to_vector(self, feature_dict, keys):
        return np.array([
            feature_dict[k] if feature_dict[k] is not None else 0.0
            for k in keys
        ], dtype=float)

    def compare(self):
        """
        Vergleicht alle CIM-ZIP-Dateien mit dem aktiven PowerFactory-Projekt.
        Gibt die euklidische Distanz pro ZIP aus und nennt das ZIP mit der kleinsten Distanz.
        """
        if self.features_cim_all is None or self.features_pf is None:
            raise RuntimeError("Features wurden noch nicht extrahiert. Bitte erst extract_features() aufrufen.")

        best_zip = None
        best_dist = float("inf")

        #Alle ZIP-Dateien durchlaufen
        for zip_name, features_cim in self.features_cim_all.items():
            keys = sorted(set(features_cim) & set(self.features_pf)) #Alle Features, die in CIM und PF vorkommen, werden in die gleiche Reihenfolge sortiert

            #Umwandlung in Vektoren, Nutzung definierte Funktion von oben 
            v_cim = self._to_vector(features_cim, keys)
            v_pf = self._to_vector(self.features_pf, keys)

            diff = v_pf - v_cim #Berechnung Differenz -> Ergebnis ist wieder ein Vektor 
            dist = np.linalg.norm(diff) #mathematische Berechnung der Länge des Vektors "diff"

            print(f"\n--- Feature-Vergleich für {zip_name} ---")
            for k, a, b, d in zip(keys, v_cim, v_pf, diff):
                print(f"{k:20s} CIM={a:8.3f} PF={b:8.3f} Δ={d:8.3f}")

            print(f"\nEuklidische Distanz für {zip_name}: {dist:.3f}")

            #Überprüfung, ob diese ZIP die kleinste Distanz hat
            if dist < best_dist:
                best_dist = dist
                best_zip = zip_name

        print(f"\nZIP-Datei mit der geringsten euklidischen Distanz: {best_zip} ({best_dist:.3f})")
