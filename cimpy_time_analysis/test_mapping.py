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


# Nach dem xml_files Sammeln:
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

def extract_scenario_time(xml_files):
    ns = {"md": "http://iec.ch/TC57/61970-552/ModelDescription/1#"}
    for f in xml_files:
        try:
            root = ET.parse(f).getroot()
            fm = root.find(".//md:FullModel", ns)
            if fm is None:
                continue
            scen = fm.findtext(".//md:Model.scenarioTime", default=None, namespaces=ns)
            if scen:
                s = scen.replace("Z", "+00:00")
                return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            pass
    return None

print("scenarioTime:", extract_scenario_time(xml_files))
