# cim_queries.py

from cim_object_utils import collect_all_cim_objects


def query_total_powerflow_over_time(cim_snapshots):
    """
    Aggregiert SvPowerFlow.p pro Snapshot (Zeitpunkt).
    """
    results = []

    for snapshot_name, cim_result in cim_snapshots.items():
        all_objects = collect_all_cim_objects(cim_result)

        sv_powerflows = [
            obj for obj in all_objects
            if obj.__class__.__name__ == "SvPowerFlow" and hasattr(obj, "p")
        ]

        if not sv_powerflows:
            continue

        total_p = sum(obj.p for obj in sv_powerflows)

        results.append({
            "snapshot": snapshot_name,
            "total_active_power": total_p
        })

    return results
