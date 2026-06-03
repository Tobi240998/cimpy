from compare_features import FeatureComparer

# -------------------------------------------------
# Start
# -------------------------------------------------
if __name__ == "__main__":
    # Ordner, in dem alle entpackten CIM-Fälle liegen
    cim_folder = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    # Name des aktiven PowerFactory-Projekts
    pf_project = "39 Bus New England System(1)"

    # FeatureComparer initialisieren
    comparer = FeatureComparer(
        cim_folder=cim_folder,
        pf_project=pf_project
    )

    # Features aus allen CIM-Fällen und dem PF-Projekt extrahieren
    comparer.extract_features()

    # Vergleich durchführen und bestes CIM-Netz ausgeben
    comparer.compare()
