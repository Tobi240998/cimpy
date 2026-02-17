from cim_object_utils import collect_all_cim_objects


def preprocess_snapshots(cim_snapshots):
    """
    Extrahiert relevante SV-Daten aus jedem Snapshot.
    """

    cache = {}

    for name, cim_result in cim_snapshots.items():

        all_objects = collect_all_cim_objects(cim_result)

        flows = [
            obj for obj in all_objects
            if obj.__class__.__name__ == "SvPowerFlow"
        ]

        cache[name] = {
            "flows": flows
        }

    return cache
