from load_cim_data import load_cim_folder
from explore_objects import summarize_object_types
from explore_objects import list_all_object_types

if __name__ == "__main__":
    cim_folder = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    cim_data = load_cim_folder(cim_folder)

    print(f"Insgesamt {len(cim_data)} XML-Dateien gefunden.\n")

    for xml_path, data in cim_data.items():
        print(f"Datei: {xml_path}")
        for obj_type, df in data.items():
            print(f" - {obj_type}: {len(df)} Objekte")
        print("-" * 60)
 

if __name__ == "__main__":
    cim_folder = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    cim_data = load_cim_folder(cim_folder)
    summarize_object_types(cim_data)
    list_all_object_types(cim_data)


