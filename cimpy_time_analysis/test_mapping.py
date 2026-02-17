import logging
import cimpy
from pathlib import Path
from cim_object_utils import collect_all_cim_objects

logging.basicConfig(filename='importCIGREMV.log', level=logging.INFO, filemode='w')

sample_folder = Path(r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted\CIM_GridAssist_1")

xml_files = [str(file.absolute()) for file in sample_folder.glob('*.xml')]
import_result = cimpy.cim_import(xml_files, "cgmes_v2_4_15")

topology = import_result["topology"]
all_objs = collect_all_cim_objects(import_result)


def canonical_id(value):
    """Normalisiert Objekt/Strings auf '_uuid' (lowercase)"""
    if value is None:
        return None
    if not isinstance(value, str):
        value = getattr(value, "mRID", None)
        if value is None:
            return None

    s = value.strip()
    if "#" in s:
        s = s.split("#")[-1]
    if s.lower().startswith("urn:uuid:"):
        s = s.split(":", 2)[-1]
    s = s.strip()
    if s and not s.startswith("_"):
        s = "_" + s
    return s.lower()


# ----------------------------
# 1) SvVoltage Index bauen
# ----------------------------
sv_voltages = [o for o in all_objs if o.__class__.__name__ == "SvVoltage"]
voltage_by_tnode = {}

for sv in sv_voltages:
    tn_ref = getattr(sv, "TopologicalNode", None)
    tn_id = canonical_id(tn_ref)
    if tn_id:
        voltage_by_tnode[tn_id] = sv

print("SvVoltage count:", len(sv_voltages))
print("voltage_by_tnode keys sample:", list(voltage_by_tnode.keys())[:5])


# ----------------------------
# 2) ConnectivityNode -> TopologicalNode Mapping bauen
# ----------------------------
cn_to_tn = {}

for o in all_objs:
    if o.__class__.__name__ == "ConnectivityNode":
        cn_id = canonical_id(getattr(o, "mRID", None))
        tn_ref = getattr(o, "TopologicalNode", None)
        tn_id = canonical_id(tn_ref)
        if cn_id and tn_id:
            cn_to_tn[cn_id] = tn_id

print("ConnectivityNode->TopologicalNode mappings:", len(cn_to_tn))
print("CN->TN sample:", list(cn_to_tn.items())[:3])


# ----------------------------
# 3) Trafo finden
# ----------------------------
target_name = "Trf 19 - 20"
trafos = [o for o in topology.values() if o.__class__.__name__ == "PowerTransformer"]

target_trafo = None
for t in trafos:
    if getattr(t, "name", None) == target_name:
        target_trafo = t
        break

if not target_trafo:
    # fallback: nimm den ersten Trafo
    target_trafo = trafos[0] if trafos else None

if not target_trafo:
    print("Kein PowerTransformer gefunden.")
    raise SystemExit

print("\nSelected Trafo:", getattr(target_trafo, "name", None), getattr(target_trafo, "mRID", None))


# --- Terminals des Trafos über Terminal.ConductingEquipment finden (statt trafo.Terminals) ---
def same_equipment(a, b):
    # a/b können Objekt oder mRID-String sein
    if a is None or b is None:
        return False
    if not isinstance(a, str):
        a = getattr(a, "mRID", None)
    if not isinstance(b, str):
        b = getattr(b, "mRID", None)
    return a is not None and b is not None and a == b


trafo_terminals = [
    o for o in all_objs
    if o.__class__.__name__ == "Terminal"
    and same_equipment(getattr(o, "ConductingEquipment", None), target_trafo)
]

print("\nTrafo terminals via Terminal.ConductingEquipment:", len(trafo_terminals))

hits = 0
for term in trafo_terminals:
    term_id = canonical_id(getattr(term, "mRID", None))
    cn_ref = getattr(term, "ConnectivityNode", None)
    cn_id = canonical_id(cn_ref)
    tn_id = cn_to_tn.get(cn_id) if cn_id else None
    sv = voltage_by_tnode.get(tn_id) if tn_id else None
    v = getattr(sv, "v", None) if sv else None

    print("\nTerminal:", term_id)
    print("  CN id:", cn_id)
    print("  TN id:", tn_id)
    print("  SvVoltage found:", bool(sv), " v:", v)

    if sv:
        hits += 1

print("\nTOTAL voltage hits for selected trafo:", hits)
