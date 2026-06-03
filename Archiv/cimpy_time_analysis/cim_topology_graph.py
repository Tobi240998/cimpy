from collections import Counter
from itertools import combinations

import networkx as nx

from cimpy.cimpy_time_analysis.cim_object_utils import collect_all_cim_objects


def _canonical_id(value):
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


def _safe_name(obj):
    if obj is None:
        return None
    return getattr(obj, "name", None)


def _safe_class(obj):
    if obj is None:
        return None
    try:
        return obj.__class__.__name__
    except Exception:
        return None


def _safe_description(obj):
    if obj is None:
        return None
    for attr in ["description", "desc", "shortName"]:
        value = getattr(obj, attr, None)
        if value:
            return value
    return None


def _iter_terminals(all_objects):
    for obj in all_objects:
        if obj.__class__.__name__ == "Terminal":
            yield obj


def _iter_connectivity_nodes(all_objects):
    for obj in all_objects:
        if obj.__class__.__name__ == "ConnectivityNode":
            yield obj


def _iter_topological_nodes(all_objects):
    for obj in all_objects:
        if obj.__class__.__name__ == "TopologicalNode":
            yield obj


def _resolve_topology_node_id_for_terminal(terminal_id, network_index, level="connectivity"):
    """
    Liefert je nach level:
    - connectivity: ConnectivityNode-ID
    - topological:  TopologicalNode-ID
    """
    terminal_to_cn = network_index.get("terminal_to_connectivitynode", {})
    cn_to_tn = network_index.get("connectivitynode_to_topologicalnode", {})

    cn_id = terminal_to_cn.get(terminal_id)
    if not cn_id:
        return None

    if level == "connectivity":
        return cn_id

    if level == "topological":
        return cn_to_tn.get(cn_id)

    raise ValueError(f"Unbekanntes level: {level!r}. Erlaubt: 'connectivity' oder 'topological'.")


def _get_equipment_node_attrs(eq_obj):
    return {
        "name": _safe_name(eq_obj),
        "cim_class": _safe_class(eq_obj),
        "kind": "equipment",
        "description": _safe_description(eq_obj),
        "mrid": getattr(eq_obj, "mRID", None),
    }


def _get_topology_node_attrs(node_obj, level):
    return {
        "name": _safe_name(node_obj),
        "cim_class": _safe_class(node_obj),
        "kind": f"{level}_node",
        "description": _safe_description(node_obj),
        "mrid": getattr(node_obj, "mRID", None),
    }


def _ensure_equipment_node(G, eq_obj):
    eq_id = _canonical_id(eq_obj)
    if not eq_id:
        return None

    if eq_id not in G:
        G.add_node(eq_id, **_get_equipment_node_attrs(eq_obj))

    return eq_id


def _ensure_named_topology_node(G, node_id, node_obj, level):
    if not node_id:
        return None

    if node_id not in G:
        if node_obj is not None:
            G.add_node(node_id, **_get_topology_node_attrs(node_obj, level))
        else:
            G.add_node(
                node_id,
                name=None,
                cim_class="ConnectivityNode" if level == "connectivity" else "TopologicalNode",
                kind=f"{level}_node",
                description=None,
                mrid=node_id,
            )

    return node_id


def _build_topology_object_lookup(first_snapshot):
    """
    Baut Lookups für echte ConnectivityNode-/TopologicalNode-Objekte,
    damit wir im Graphen brauchbare Metadaten an die Knoten hängen können.
    """
    all_objects = collect_all_cim_objects(first_snapshot)

    connectivity_lookup = {}
    topological_lookup = {}

    for obj in _iter_connectivity_nodes(all_objects):
        obj_id = _canonical_id(obj)
        if obj_id:
            connectivity_lookup[obj_id] = obj

    for obj in _iter_topological_nodes(all_objects):
        obj_id = _canonical_id(obj)
        if obj_id:
            topological_lookup[obj_id] = obj

    return connectivity_lookup, topological_lookup


def build_cim_topology_graph(
    first_snapshot,
    network_index,
    level="connectivity",
    include_topology_nodes=False,
):
    """
    Baut einen MultiGraph aus CIM-Terminals und den im network_index hinterlegten Mappings.

    Standardverhalten:
    - Knoten = Equipments
    - Kanten = gemeinsame ConnectivityNode bzw. TopologicalNode

    Optional:
    - include_topology_nodes=True
      -> zusätzlich bipartiter Graph: Equipment -- ConnectivityNode/TopologicalNode

    Parameters
    ----------
    first_snapshot : dict
        Erster importierter CIM-Snapshot.
    network_index : dict
        Index aus build_network_index(...).
    level : str
        "connectivity" oder "topological"
    include_topology_nodes : bool
        Wenn True, werden die Connectivity-/TopologicalNodes als explizite Knoten
        in den Graphen aufgenommen.

    Returns
    -------
    nx.MultiGraph
    """
    if level not in {"connectivity", "topological"}:
        raise ValueError("level muss 'connectivity' oder 'topological' sein.")

    G = nx.MultiGraph()
    G.graph["model"] = "cim_topology"
    G.graph["level"] = level
    G.graph["include_topology_nodes"] = include_topology_nodes

    terminals_to_equipment = network_index.get("terminals_to_equipment", {})
    equipment_to_terminal_ids = network_index.get("equipment_to_terminal_ids", {})

    connectivity_lookup, topological_lookup = _build_topology_object_lookup(first_snapshot)

    # ---------------------------------------------------------
    # 1) Gruppierung: topology_node_id -> Liste von Anschlussinfos
    # ---------------------------------------------------------
    grouped = {}

    for equipment_id, terminal_ids in equipment_to_terminal_ids.items():
        eq_obj = None

        for terminal_id in terminal_ids:
            if terminal_id in terminals_to_equipment:
                eq_obj = terminals_to_equipment[terminal_id]
                break

        if eq_obj is None:
            continue

        eq_node_id = _ensure_equipment_node(G, eq_obj)
        if not eq_node_id:
            continue

        for terminal_id in terminal_ids:
            topology_node_id = _resolve_topology_node_id_for_terminal(
                terminal_id=terminal_id,
                network_index=network_index,
                level=level,
            )
            if not topology_node_id:
                continue

            grouped.setdefault(topology_node_id, []).append(
                {
                    "equipment_id": eq_node_id,
                    "equipment_obj": eq_obj,
                    "terminal_id": terminal_id,
                }
            )

    # ---------------------------------------------------------
    # 2) Optional: topology nodes explizit anlegen + Equipment anbinden
    # ---------------------------------------------------------
    if include_topology_nodes:
        for topology_node_id, entries in grouped.items():
            topo_obj = None
            if level == "connectivity":
                topo_obj = connectivity_lookup.get(topology_node_id)
            elif level == "topological":
                topo_obj = topological_lookup.get(topology_node_id)

            _ensure_named_topology_node(G, topology_node_id, topo_obj, level)

            for entry in entries:
                eq_id = entry["equipment_id"]
                terminal_id = entry["terminal_id"]

                edge_key = f"{level}_membership::{eq_id}::{topology_node_id}::{terminal_id}"
                if G.has_edge(eq_id, topology_node_id, key=edge_key):
                    continue

                G.add_edge(
                    eq_id,
                    topology_node_id,
                    key=edge_key,
                    type=f"{level}_membership",
                    level=level,
                    terminal_id=terminal_id,
                    topology_node_id=topology_node_id,
                )

    # ---------------------------------------------------------
    # 3) Equipment-Equipment-Kanten über gemeinsamen Topologieknoten
    # ---------------------------------------------------------
    for topology_node_id, entries in grouped.items():
        if len(entries) < 2:
            continue

        for left, right in combinations(entries, 2):
            u = left["equipment_id"]
            v = right["equipment_id"]

            if not u or not v or u == v:
                continue

            u_obj = left["equipment_obj"]
            v_obj = right["equipment_obj"]

            edge_key = (
                f"{level}_shared::{topology_node_id}::"
                f"{left['terminal_id']}::{right['terminal_id']}::{u}::{v}"
            )

            G.add_edge(
                u,
                v,
                key=edge_key,
                type=f"shared_{level}_node",
                level=level,
                topology_node_id=topology_node_id,
                terminal_1_id=left["terminal_id"],
                terminal_2_id=right["terminal_id"],
                equipment_1_name=_safe_name(u_obj),
                equipment_1_class=_safe_class(u_obj),
                equipment_2_name=_safe_name(v_obj),
                equipment_2_class=_safe_class(v_obj),
            )

    return G


def summarize_graph_basic(G):
    """
    Liefert einfache Kennzahlen für Debugging / spätere Antwortgenerierung.
    """
    if G is None:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "num_connected_components": 0,
            "largest_component_size": 0,
            "degree_distribution_top10": [],
            "node_kind_counts": {},
            "edge_type_counts": {},
            "level": None,
        }

    node_kind_counts = Counter()
    for _, data in G.nodes(data=True):
        node_kind_counts[data.get("kind", "unknown")] += 1

    edge_type_counts = Counter()
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        for _, _, _, data in G.edges(keys=True, data=True):
            edge_type_counts[data.get("type", "unknown")] += 1
    else:
        for _, _, data in G.edges(data=True):
            edge_type_counts[data.get("type", "unknown")] += 1

    comps = list(nx.connected_components(G)) if G.number_of_nodes() else []
    largest_component_size = max((len(c) for c in comps), default=0)

    degree_distribution_top10 = Counter(dict(G.degree()).values()).most_common(10)

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_connected_components": len(comps),
        "largest_component_size": largest_component_size,
        "degree_distribution_top10": degree_distribution_top10,
        "node_kind_counts": dict(node_kind_counts),
        "edge_type_counts": dict(edge_type_counts),
        "level": G.graph.get("level"),
    }


def get_equipment_node_id(equipment_obj):
    return _canonical_id(equipment_obj)


def get_equipment_neighbors(G, equipment_obj_or_id, allowed_neighbor_classes=None):
    """
    Liefert direkte Nachbarn eines Equipments im Graphen.

    Returns
    -------
    list[dict]
        Jeder Eintrag enthält u.a.:
        - equipment_id
        - name
        - cim_class
        - degree
        - edge_types
        - shared_topology_node_ids
    """
    eq_id = _canonical_id(equipment_obj_or_id)
    if not eq_id or G is None or eq_id not in G:
        return []

    neighbors = []

    for nbr in G.neighbors(eq_id):
        node_data = G.nodes.get(nbr, {})
        nbr_class = node_data.get("cim_class")
        nbr_kind = node_data.get("kind")

        if nbr_kind != "equipment":
            continue

        if allowed_neighbor_classes and nbr_class not in allowed_neighbor_classes:
            continue

        shared_topology_node_ids = set()
        edge_types = Counter()

        edge_bundle = G.get_edge_data(eq_id, nbr) or {}
        for _, ed in edge_bundle.items():
            edge_types[ed.get("type", "unknown")] += 1
            topo_id = ed.get("topology_node_id")
            if topo_id:
                shared_topology_node_ids.add(topo_id)

        neighbors.append(
            {
                "equipment_id": nbr,
                "name": node_data.get("name"),
                "cim_class": nbr_class,
                "degree": G.degree(nbr),
                "edge_count": len(edge_bundle),
                "edge_types": dict(edge_types),
                "shared_topology_node_ids": sorted(shared_topology_node_ids),
            }
        )

    neighbors.sort(key=lambda x: ((x["name"] or ""), x["equipment_id"]))
    return neighbors


def get_connected_component_for_equipment(G, equipment_obj_or_id):
    """
    Liefert die gesamte zusammenhängende Komponente eines Equipments.
    """
    eq_id = _canonical_id(equipment_obj_or_id)
    if not eq_id or G is None or eq_id not in G:
        return {
            "equipment_id": eq_id,
            "component_node_ids": [],
            "component_size": 0,
            "equipment_nodes": [],
            "node_kind_counts": {},
            "contains_equipment": False,
        }

    comp_nodes = None
    for comp in nx.connected_components(G):
        if eq_id in comp:
            comp_nodes = comp
            break

    if comp_nodes is None:
        return {
            "equipment_id": eq_id,
            "component_node_ids": [],
            "component_size": 0,
            "equipment_nodes": [],
            "node_kind_counts": {},
            "contains_equipment": False,
        }

    node_kind_counts = Counter()
    equipment_nodes = []

    for node_id in sorted(comp_nodes):
        data = G.nodes.get(node_id, {})
        kind = data.get("kind", "unknown")
        node_kind_counts[kind] += 1

        if kind == "equipment":
            equipment_nodes.append(
                {
                    "equipment_id": node_id,
                    "name": data.get("name"),
                    "cim_class": data.get("cim_class"),
                    "degree": G.degree(node_id),
                }
            )

    equipment_nodes.sort(key=lambda x: ((x["name"] or ""), x["equipment_id"]))

    return {
        "equipment_id": eq_id,
        "component_node_ids": sorted(comp_nodes),
        "component_size": len(comp_nodes),
        "equipment_nodes": equipment_nodes,
        "node_kind_counts": dict(node_kind_counts),
        "contains_equipment": True,
    }


def find_shortest_path_between_equipments(G, source_equipment_obj_or_id, target_equipment_obj_or_id):
    """
    Kürzester Pfad zwischen zwei Equipments.
    .
    """
    source_id = _canonical_id(source_equipment_obj_or_id)
    target_id = _canonical_id(target_equipment_obj_or_id)

    if not source_id or not target_id:
        return {
            "found": False,
            "reason": "missing_id",
            "path": [],
            "path_details": [],
        }

    if G is None or source_id not in G or target_id not in G:
        return {
            "found": False,
            "reason": "node_not_in_graph",
            "path": [],
            "path_details": [],
        }

    try:
        path = nx.shortest_path(G, source=source_id, target=target_id)
    except nx.NetworkXNoPath:
        return {
            "found": False,
            "reason": "no_path",
            "path": [],
            "path_details": [],
        }

    path_details = []
    for node_id in path:
        data = G.nodes.get(node_id, {})
        path_details.append(
            {
                "node_id": node_id,
                "name": data.get("name"),
                "cim_class": data.get("cim_class"),
                "kind": data.get("kind"),
                "degree": G.degree(node_id),
            }
        )

    return {
        "found": True,
        "reason": None,
        "path": path,
        "path_details": path_details,
        "path_length": max(0, len(path) - 1),
    }


def build_topology_debug_report(first_snapshot, network_index):
    """
    Kleiner Debug-Report für schnellen Konsolencheck.
    Baut beide Graphen:
    - connectivity
    - topological
    """
    G_connectivity = build_cim_topology_graph(
        first_snapshot=first_snapshot,
        network_index=network_index,
        level="connectivity",
        include_topology_nodes=False,
    )

    G_topological = build_cim_topology_graph(
        first_snapshot=first_snapshot,
        network_index=network_index,
        level="topological",
        include_topology_nodes=False,
    )

    return {
        "connectivity_graph": summarize_graph_basic(G_connectivity),
        "topological_graph": summarize_graph_basic(G_topological),
    }