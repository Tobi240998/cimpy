from pathlib import Path
from cimpy.cimimport import cim_import

# Pfad zu den CIM-Dateien
case_path = Path(r"C:\Users\STELLER\Documents\Masterarbeit\Test CIM\examples\case1")
files = [str(p) for p in case_path.glob("*.xml")]

if not files:
    print("Keine CIM-Dateien gefunden!")
    exit(1)

print("Gefundene CIM-Dateien:")
for f in files:
    print(" -", f)

# CIM importieren
grid = cim_import(files, cgmes_version="2.4.15")

print(grid.keys())

print("\nCIM erfolgreich geladen!")
print("Verfügbare Klassen im Grid:")
print(list(grid.keys()))  # statt .classes()

# EnergyConsumer / Loads ausgeben
loads = grid.get("EnergyConsumer", [])
print(f"\nGefundene Loads (EnergyConsumer): {len(loads)}")

for load in loads[:10]:  # nur die ersten 10
    p = getattr(load, "p", "N/A")
    q = getattr(load, "q", "N/A")
    print(f" - {load.mRID}: p={p}, q={q}")
