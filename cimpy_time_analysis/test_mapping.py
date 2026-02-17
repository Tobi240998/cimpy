import logging
import cimpy
from pathlib import Path
import math
from collections import defaultdict

logging.basicConfig(filename='importCIGREMV.log', level=logging.INFO, filemode='w')

sample_folder = Path(r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted\CIM_GridAssist_1")

xml_files = []
for file in sample_folder.glob('*.xml'):
    xml_files.append(str(file.absolute()))

import_result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

topology = import_result['topology']

# Dictionary: Trafo → maximale Scheinleistung
trafo_loading = defaultdict(float)

# Dictionary: Trafo → Nennleistung
trafo_rated = {}

for element in topology.values():

    if element.__class__.__name__ == "SvPowerFlow":

        terminal = element.Terminal
        if terminal is None:
            continue

        equipment = terminal.ConductingEquipment
        if equipment is None:
            continue

        # Nur Trafos betrachten
        if equipment.__class__.__name__ == "PowerTransformer":

            # S berechnen
            p = element.p
            q = element.q
            s = math.sqrt(p**2 + q**2)

            # Maximum pro Trafo merken
            trafo_loading[equipment] = max(trafo_loading[equipment], s)

            # Nennleistung speichern (nur einmal)
            if equipment not in trafo_rated:
                if equipment.PowerTransformerEnd:
                    trafo_rated[equipment] = equipment.PowerTransformerEnd[0].ratedS


# ---- Ausgabe ----
for trafo, s_actual in trafo_loading.items():

    rated_s = trafo_rated.get(trafo, 0)

    if rated_s == 0:
        print(f"\nTrafo: {trafo.name}")
        print("  Keine Nennleistung gefunden.")
        continue

    loading = s_actual / rated_s * 100

    print(f"\nTrafo: {trafo.name}")
    print(f"  Nennleistung: {rated_s:.2f} MVA")
    print(f"  Maximale Leistung: {s_actual:.2f} MVA")
    print(f"  Auslastung: {loading:.2f} %")

