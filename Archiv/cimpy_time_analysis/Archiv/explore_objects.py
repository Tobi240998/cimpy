from collections import defaultdict

def summarize_object_types(cim_data):
    """
    Erstellt eine globale Übersicht über alle CIM-Objekttypen.
    """
    object_summary = defaultdict(int)
    occurrence_by_case = defaultdict(set)

    for case_name, case_data in cim_data.items():
        for obj_type, df in case_data.items():
            object_summary[obj_type] += len(df)
            occurrence_by_case[obj_type].add(case_name)

    print("\n===== Globale Objektübersicht =====\n")

    for obj_type in sorted(object_summary.keys()):
        total_objects = object_summary[obj_type]
        num_cases = len(occurrence_by_case[obj_type])

        print(f"{obj_type}")
        print(f"   Gesamtanzahl Objekte: {total_objects}")
        print(f"   Anzahl Zeitpunkte:    {num_cases}")
        print("-" * 40)


def list_all_object_types(cim_data):
    """
    Gibt eine eindeutige Liste aller CIM-Objekttypen aus,
    die irgendwo im Datensatz vorkommen.
    """
    object_types = set()

    for case_data in cim_data.values():
        object_types.update(case_data.keys())

    print("\n===== CIM-Objekttypen im Netz =====\n")
    for obj_type in sorted(object_types):
        print(obj_type)

    print(f"\nGesamtanzahl unterschiedlicher Objekttypen: {len(object_types)}")
