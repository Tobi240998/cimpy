# cim_queries.py

from cim_object_utils import collect_all_cim_objects


def query_total_powerflow_over_time(snapshot_cache):
    results = []

    for snapshot, data in snapshot_cache.items():
        flows = data.get("SvPowerFlow", [])
        if not flows:
            continue

        total_p = sum(obj.p for obj in flows)

        results.append({
            "snapshot": snapshot,
            "total_active_power": total_p
        })

    return results

def summarize_powerflow(results):
    values = [r["total_active_power"] for r in results]

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "peak_snapshot": max(results, key=lambda r: r["total_active_power"])["snapshot"]
    }

