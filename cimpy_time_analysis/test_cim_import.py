import cimpy
import os
from cim_object_utils import collect_all_cim_objects

# Pfad zu EINEM konkreten CIM-Snapshot-Ordner
cim_case_folder = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted\CIM_GridAssist_8"

# XML-Dateien sammeln
xml_files = [
    os.path.join(cim_case_folder, f)
    for f in os.listdir(cim_case_folder)
    if f.lower().endswith(".xml")
]

print(f"Gefundene XML-Dateien: {len(xml_files)}")

# -------------------------------------------------
# CIMpy Import
# -------------------------------------------------
result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

print("\n=== Rückgabetyp von cim_import ===")
print(type(result))

print("\n=== Keys im Rückgabeobjekt ===")
try:
    print(result.keys())
except Exception as e:
    print("Kein dict:", e)

print("\n=== Typen der einzelnen Einträge ===")
if isinstance(result, dict):
    for key, value in result.items():
        print(f"{key}: {type(value)}")

# -------------------------------------------------
# Suche nach CIM-Objekten (SvVoltage, SvPowerFlow)
# -------------------------------------------------
print("\n=== Suche nach Sv*-Objekten ===")

found_sv_voltage = False
found_sv_powerflow = False

def inspect_container(container):
    global found_sv_voltage, found_sv_powerflow

    if isinstance(container, dict):
        for v in container.values():
            inspect_container(v)
    elif isinstance(container, list):
        for v in container:
            inspect_container(v)
    else:
        cls_name = container.__class__.__name__
        if cls_name == "SvVoltage":
            found_sv_voltage = True
        if cls_name == "SvPowerFlow":
            found_sv_powerflow = True

inspect_container(result)

print(f"SvVoltage gefunden: {found_sv_voltage}")
print(f"SvPowerFlow gefunden: {found_sv_powerflow}")


all_objects = collect_all_cim_objects(result)

sv_voltages = [obj for obj in all_objects if obj.__class__.__name__ == "SvVoltage"]
sv_powerflows = [obj for obj in all_objects if obj.__class__.__name__ == "SvPowerFlow"]

print(f"Anzahl SvVoltage: {len(sv_voltages)}")
print(f"Anzahl SvPowerFlow: {len(sv_powerflows)}")

# Test: Werte auslesen
if sv_voltages:
    print("Beispiel Spannung:", sv_voltages[0].v)

if sv_powerflows:
    print("Beispiel Leistung:", sv_powerflows[0].p)
