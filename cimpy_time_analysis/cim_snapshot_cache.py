from cim_object_utils import collect_all_cim_objects
def preprocess_snapshots(cim_snapshots):
    cache = {}

    for name, cim_result in cim_snapshots.items():

        all_objects = collect_all_cim_objects(cim_result)

        flows = [
            obj for obj in all_objects
            if obj.__class__.__name__ == "SvPowerFlow"
        ]

        trafos = [
            obj for obj in all_objects
            if obj.__class__.__name__ == "PowerTransformer"
        ]

        cache[name] = {
            "flows": flows,
            "trafos": trafos
        }

    return cache


