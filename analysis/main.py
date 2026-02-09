from feature_extract_pf import PFFeatureExtractor #Import der Klasse zum Feature Extract von PowerFactory
from feature_extract_cim import CIMFeatureExtractor #Import der Klasse zum Feature Extract von CIM
from compare_features import FeatureComparer

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