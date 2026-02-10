from cim_object_utils import collect_all_cim_objects


def preprocess_snapshots(cim_snapshots):
    """
    Extrahiert relevante StateVariables einmalig pro Snapshot.
    """
    cache = {}

    for name, cim_result in cim_snapshots.items():
        all_objects = collect_all_cim_objects(cim_result)

        cache[name] = {
            "SvVoltage": [
                obj for obj in all_objects
                if obj.__class__.__name__ == "SvVoltage" and hasattr(obj, "v")
            ],
            "SvPowerFlow": [
                obj for obj in all_objects
                if obj.__class__.__name__ == "SvPowerFlow" and hasattr(obj, "p")
            ]
        }

    return cache
