from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# ------------------------------------------------------------------
# SAFE HELPERS
# ------------------------------------------------------------------
def safe_fullname(obj: Any) -> Optional[str]:
    try:
        return obj.GetFullName()
    except Exception:
        try:
            return obj.loc_name
        except Exception:
            return None


def safe_class(obj: Any) -> Optional[str]:
    try:
        return obj.GetClassName()
    except Exception:
        return None


def safe_name(obj: Any) -> Optional[str]:
    try:
        return obj.loc_name
    except Exception:
        return None


# ------------------------------------------------------------------
# GRAPH HELPERS
# ------------------------------------------------------------------
def add_node(
    graph: nx.MultiGraph,
    obj: Any,
    kind: Optional[str] = None,
) -> Optional[str]:
    node_id = safe_fullname(obj)
    if not node_id:
        return None

    if node_id not in graph:
        graph.add_node(
            node_id,
            name=safe_name(obj),
            pf_class=safe_class(obj),
            full_name=node_id,
            kind=kind,
        )

    return node_id


def add_edge(
    graph: nx.MultiGraph,
    a_obj: Any,
    b_obj: Any,
    etype: str,
    edge_obj: Any | None = None,
) -> bool:
    a = add_node(graph, a_obj)
    b = add_node(graph, b_obj)

    if not a or not b:
        return False

    edge_full_name = safe_fullname(edge_obj) if edge_obj is not None else None

    graph.add_edge(
        a,
        b,
        key=edge_full_name if edge_full_name else f"{a}__{b}__{etype}",
        type=etype,
        edge_name=safe_name(edge_obj) if edge_obj is not None else None,
        edge_class=safe_class(edge_obj) if edge_obj is not None else None,
        edge_full_name=edge_full_name,
    )
    return True


def is_cubic_node(node_id: str, data: Dict[str, Any]) -> bool:
    return data.get("pf_class") == "StaCubic" or data.get("kind") == "cubic"


# ------------------------------------------------------------------
# BUILD RAW / CONTRACTED GRAPHS
# ------------------------------------------------------------------
def build_wiring_graph(app: Any) -> Tuple[nx.MultiGraph, Dict[str, Any]]:
    graph = nx.MultiGraph()

    debug: Dict[str, Any] = {
        "counts": {
            "cubics_total": 0,
            "cubics_with_terminal": 0,
            "stas_total": 0,
            "stas_with_obj_id": 0,
            "elms_total": 0,
            "elms_line_like": 0,
            "elms_transformer_like": 0,
        },
        "samples": {
            "cubic_names": [],
            "sta_names": [],
            "elm_names": [],
            "node_name_samples": [],
        },
    }

    cubics = app.GetCalcRelevantObjects("*.StaCubic") or []
    debug["counts"]["cubics_total"] = len(cubics)
    debug["samples"]["cubic_names"] = [safe_name(x) for x in cubics[:10]]

    for cubic in cubics:
        try:
            terminal = getattr(cubic, "cterm", None)
            if terminal:
                debug["counts"]["cubics_with_terminal"] += 1
                add_node(graph, cubic, kind="cubic")
                add_node(graph, terminal, kind="terminal")
                add_edge(graph, terminal, cubic, etype="cterm_cubic", edge_obj=cubic)
        except Exception:
            continue

    stas = app.GetCalcRelevantObjects("*.Sta*") or []
    debug["counts"]["stas_total"] = len(stas)
    debug["samples"]["sta_names"] = [safe_name(x) for x in stas[:10]]

    for sta in stas:
        try:
            obj_id = getattr(sta, "obj_id", None)
            if obj_id:
                debug["counts"]["stas_with_obj_id"] += 1
                add_node(graph, sta, kind="sta")
                add_node(graph, obj_id, kind="equip")
                add_edge(graph, sta, obj_id, etype="sta_objid", edge_obj=sta)
        except Exception:
            continue

    elms = app.GetCalcRelevantObjects("*.Elm*") or []
    debug["counts"]["elms_total"] = len(elms)
    debug["samples"]["elm_names"] = [safe_name(x) for x in elms[:10]]

    for elm in elms:
        try:
            bus1 = getattr(elm, "bus1", None)
            bus2 = getattr(elm, "bus2", None)
            if bus1 and bus2:
                debug["counts"]["elms_line_like"] += 1
                add_node(graph, elm, kind="equip")
                add_node(graph, bus1, kind="endpoint")
                add_node(graph, bus2, kind="endpoint")
                add_edge(graph, bus1, elm, etype="end1_equip", edge_obj=elm)
                add_edge(graph, bus2, elm, etype="end2_equip", edge_obj=elm)
        except Exception:
            pass

        try:
            bushv = getattr(elm, "bushv", None)
            buslv = getattr(elm, "buslv", None)
            if bushv and buslv:
                debug["counts"]["elms_transformer_like"] += 1
                add_node(graph, elm, kind="equip")
                add_node(graph, bushv, kind="endpoint")
                add_node(graph, buslv, kind="endpoint")
                add_edge(graph, bushv, elm, etype="hv_equip", edge_obj=elm)
                add_edge(graph, buslv, elm, etype="lv_equip", edge_obj=elm)
        except Exception:
            pass

    debug["samples"]["node_name_samples"] = [
        {
            "node_id": node_id,
            "name": data.get("name"),
            "pf_class": data.get("pf_class"),
            "full_name": data.get("full_name"),
            "kind": data.get("kind"),
        }
        for node_id, data in list(graph.nodes(data=True))[:25]
    ]

    return graph, debug


def contract_cubicles_as_edges(wiring_graph: nx.MultiGraph) -> nx.MultiGraph:
    contracted = nx.MultiGraph()
    contracted.add_nodes_from(wiring_graph.nodes(data=True))

    node_data = dict(wiring_graph.nodes(data=True))

    for node_id, data in wiring_graph.nodes(data=True):
        if not is_cubic_node(node_id, data):
            continue

        neighbors = list(wiring_graph.neighbors(node_id))
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u = neighbors[i]
                v = neighbors[j]
                contracted.add_edge(
                    u,
                    v,
                    key=f"via_cubic::{node_id}::{u}::{v}",
                    type="via_cubic",
                    cubic_full_name=node_id,
                    cubic_name=node_data[node_id].get("name"),
                    cubic_class=node_data[node_id].get("pf_class"),
                )

    for u, v, key, edge_data in wiring_graph.edges(keys=True, data=True):
        if is_cubic_node(u, node_data.get(u, {})) or is_cubic_node(v, node_data.get(v, {})):
            continue
        contracted.add_edge(u, v, key=f"orig::{key}", **edge_data)

    cubic_nodes = [n for n, d in wiring_graph.nodes(data=True) if is_cubic_node(n, d)]
    contracted.remove_nodes_from([n for n in cubic_nodes if n in contracted])

    return contracted


# ------------------------------------------------------------------
# GRAPH ANALYSIS / INVENTORY
# ------------------------------------------------------------------
def summarize_graph(graph: nx.MultiGraph) -> Dict[str, Any]:
    components = list(nx.connected_components(graph))
    component_sizes = sorted((len(c) for c in components), reverse=True)

    degree_map = dict(graph.degree())
    degree_distribution = Counter(degree_map.values())

    class_counter = Counter()
    for _, data in graph.nodes(data=True):
        pf_class = data.get("pf_class") or "<unknown>"
        class_counter[pf_class] += 1

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_connected_components": len(components),
        "largest_component_sizes": component_sizes[:10],
        "degree_distribution_top10": degree_distribution.most_common(10),
        "node_classes_top20": class_counter.most_common(20),
    }


def classify_inventory_type(node_attrs: Dict[str, Any]) -> str:
    pf_class = (node_attrs.get("pf_class") or "").lower()
    kind = (node_attrs.get("kind") or "").lower()

    if pf_class == "elmterm" or kind == "terminal":
        return "bus"
    if pf_class in {"elmlod", "elmlodlv", "elmlodmv"}:
        return "load"
    if "tr" in pf_class and pf_class.startswith("elmtr"):
        return "transformer"
    if pf_class in {"elmlne", "elmcable"} or "line" in pf_class or "cable" in pf_class:
        return "line"
    if pf_class in {"elmsym", "elmasm", "elmsgen", "elmpvsys"}:
        return "generator"
    if "coup" in pf_class or "switch" in pf_class or pf_class in {"relfuse", "staswit", "elmcoup"}:
        return "switch"
    if kind == "equip":
        return "equipment"
    if kind == "sta":
        return "station_object"
    return "unknown"


def build_topology_inventory(graph: nx.MultiGraph) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for node_id, attrs in graph.nodes(data=True):
        entry = {
            "node_id": node_id,
            "name": attrs.get("name"),
            "pf_class": attrs.get("pf_class"),
            "full_name": attrs.get("full_name"),
            "kind": attrs.get("kind"),
            "degree": graph.degree(node_id),
            "inventory_type": classify_inventory_type(attrs),
        }
        inv_type = entry["inventory_type"]
        by_type[inv_type].append(entry)

    for inv_type in by_type:
        by_type[inv_type].sort(
            key=lambda item: (
                str(item.get("name") or ""),
                str(item.get("full_name") or ""),
            )
        )

    counts = {key: len(value) for key, value in sorted(by_type.items(), key=lambda kv: kv[0])}
    samples = {
        key: [
            {
                "name": item.get("name"),
                "pf_class": item.get("pf_class"),
                "full_name": item.get("full_name"),
            }
            for item in value[:10]
        ]
        for key, value in by_type.items()
    }

    return {
        "available_types": sorted(by_type.keys()),
        "counts_by_type": counts,
        "items_by_type": dict(by_type),
        "samples_by_type": samples,
    }


# ------------------------------------------------------------------
# MATCHING
# ------------------------------------------------------------------
def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _tokenize(value: str) -> List[str]:
    text = _normalize_text(value)
    for ch in "\\/()[]{}:;,.!?\"'":
        text = text.replace(ch, " ")
    return [token for token in text.split() if token]


def _score_candidate_against_query(
    query: str,
    candidate_name: str,
    candidate_full_name: str,
) -> int:
    q = _normalize_text(query)
    name = _normalize_text(candidate_name)
    full_name = _normalize_text(candidate_full_name)

    if not q:
        return 0

    score = 0

    if q == name:
        score = max(score, 150)
    if q == full_name:
        score = max(score, 145)
    if q in name:
        score = max(score, 110)
    if q in full_name:
        score = max(score, 95)

    q_tokens = _tokenize(q)
    name_tokens = set(_tokenize(name))
    full_tokens = set(_tokenize(full_name))

    if q_tokens:
        overlap_name = sum(1 for token in q_tokens if token in name_tokens)
        overlap_full = sum(1 for token in q_tokens if token in full_tokens)
        score += 20 * overlap_name
        score += 8 * overlap_full

        joined = " ".join(q_tokens)
        if joined and joined in name:
            score += 30
        if joined and joined in full_name:
            score += 15

    return score


def find_matching_nodes(
    graph: nx.MultiGraph,
    asset_query: str,
    max_results: int = 10,
    class_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = (asset_query or "").strip().lower()
    if not query:
        return []

    matches: List[Dict[str, Any]] = []

    for node_id, data in graph.nodes(data=True):
        name = str(data.get("name") or "")
        full_name = str(data.get("full_name") or node_id)
        pf_class = str(data.get("pf_class") or "")
        kind = str(data.get("kind") or "")

        score = _score_candidate_against_query(query, name, full_name)
        if score <= 0:
            continue

        inv_type = classify_inventory_type(data)
        if class_hint:
            if inv_type == class_hint:
                score += 25
            else:
                score -= 10

        matches.append({
            "node_id": node_id,
            "name": data.get("name"),
            "pf_class": data.get("pf_class"),
            "full_name": full_name,
            "kind": kind,
            "score": score,
            "degree": graph.degree(node_id),
            "inventory_type": inv_type,
        })

    matches.sort(
        key=lambda item: (
            -item["score"],
            -(item.get("degree") or 0),
            str(item.get("full_name") or ""),
        )
    )
    return matches[:max_results]


def find_matches_in_inventory(
    inventory_items: List[Dict[str, Any]],
    raw_query: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    query_variants: List[str] = []
    raw = (raw_query or "").strip()
    if raw:
        query_variants.append(raw)

    tokens = _tokenize(raw)
    if len(tokens) >= 2:
        for window_size in range(len(tokens), 0, -1):
            for start in range(0, len(tokens) - window_size + 1):
                variant = " ".join(tokens[start:start + window_size]).strip()
                if variant and variant not in query_variants:
                    query_variants.append(variant)

    scored: List[Dict[str, Any]] = []

    for item in inventory_items:
        best_score = 0
        best_variant = None

        candidate_name = item.get("name") or ""
        candidate_full_name = item.get("full_name") or ""

        for variant in query_variants:
            score = _score_candidate_against_query(
                query=variant,
                candidate_name=candidate_name,
                candidate_full_name=candidate_full_name,
            )
            if score > best_score:
                best_score = score
                best_variant = variant

        if best_score > 0:
            entry = dict(item)
            entry["score"] = best_score
            entry["matched_query_variant"] = best_variant
            scored.append(entry)

    scored.sort(
        key=lambda item: (
            -item["score"],
            -(item.get("degree") or 0),
            str(item.get("name") or ""),
            str(item.get("full_name") or ""),
        )
    )
    return scored[:max_results]


# ------------------------------------------------------------------
# NEIGHBOR PAYLOAD
# ------------------------------------------------------------------
def get_neighbor_payload(graph: nx.MultiGraph, node_id: str) -> Dict[str, Any]:
    if node_id not in graph:
        return {
            "status": "error",
            "error": "node_not_found",
            "answer": f"Knoten nicht im Topologiegraph gefunden: {node_id}",
        }

    node_attrs = graph.nodes[node_id]
    neighbors: List[Dict[str, Any]] = []

    for neighbor_id in graph.neighbors(node_id):
        neighbor_attrs = graph.nodes[neighbor_id]
        edge_map = graph.get_edge_data(node_id, neighbor_id) or {}

        edges: List[Dict[str, Any]] = []
        for _, edge_data in edge_map.items():
            edges.append({
                "type": edge_data.get("type"),
                "edge_name": edge_data.get("edge_name"),
                "edge_class": edge_data.get("edge_class"),
                "edge_full_name": edge_data.get("edge_full_name"),
                "cubic_name": edge_data.get("cubic_name"),
                "cubic_class": edge_data.get("cubic_class"),
                "cubic_full_name": edge_data.get("cubic_full_name"),
            })

        neighbors.append({
            "node_id": neighbor_id,
            "name": neighbor_attrs.get("name"),
            "pf_class": neighbor_attrs.get("pf_class"),
            "full_name": neighbor_attrs.get("full_name"),
            "kind": neighbor_attrs.get("kind"),
            "degree": graph.degree(neighbor_id),
            "inventory_type": classify_inventory_type(neighbor_attrs),
            "edges": edges,
        })

    neighbors.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("full_name") or "")))

    return {
        "status": "ok",
        "selected_node": {
            "node_id": node_id,
            "name": node_attrs.get("name"),
            "pf_class": node_attrs.get("pf_class"),
            "full_name": node_attrs.get("full_name"),
            "kind": node_attrs.get("kind"),
            "degree": graph.degree(node_id),
            "inventory_type": classify_inventory_type(node_attrs),
        },
        "neighbors": neighbors,
        "neighbor_count": len(neighbors),
    }


# ------------------------------------------------------------------
# SERVICE-LAYER ENTRY FUNCTIONS
# ------------------------------------------------------------------
def build_powerfactory_topology_graph_from_services(
    services: Dict[str, Any],
    contract_cubicles: bool = True,
) -> Dict[str, Any]:
    app = services["app"]
    project_name = services.get("project_name")

    wiring_graph, build_debug = build_wiring_graph(app)
    analysis_graph = contract_cubicles_as_edges(wiring_graph) if contract_cubicles else wiring_graph
    inventory = build_topology_inventory(analysis_graph)

    build_debug["analysis_graph"] = {
        "graph_mode": "contracted" if contract_cubicles else "wiring",
        "num_nodes": analysis_graph.number_of_nodes(),
        "num_edges": analysis_graph.number_of_edges(),
    }

    return {
        "status": "ok",
        "tool": "build_topology_graph",
        "project": project_name,
        "graph_mode": "contracted" if contract_cubicles else "wiring",
        "wiring_graph": wiring_graph,
        "topology_graph": analysis_graph,
        "graph_summary": summarize_graph(analysis_graph),
        "inventory": inventory,
        "build_debug": build_debug,
    }


def query_powerfactory_topology_neighbors_from_services(
    services: Dict[str, Any],
    topology_graph: nx.MultiGraph,
    asset_query: str,
    selected_node_id: str | None = None,
    matches: List[Dict[str, Any]] | None = None,
    max_matches: int = 10,
) -> Dict[str, Any]:
    project_name = services.get("project_name")

    if topology_graph is None:
        return {
            "status": "error",
            "tool": "query_topology_neighbors",
            "project": project_name,
            "asset_query": asset_query,
            "error": "missing_topology_graph",
            "answer": "Es wurde kein Topologiegraph übergeben.",
        }

    if selected_node_id:
        selected = None
        if matches:
            for item in matches:
                if item.get("node_id") == selected_node_id:
                    selected = item
                    break

        if selected is None and selected_node_id in topology_graph:
            attrs = topology_graph.nodes[selected_node_id]
            selected = {
                "node_id": selected_node_id,
                "name": attrs.get("name"),
                "pf_class": attrs.get("pf_class"),
                "full_name": attrs.get("full_name"),
                "kind": attrs.get("kind"),
                "degree": topology_graph.degree(selected_node_id),
                "inventory_type": classify_inventory_type(attrs),
            }

        if selected is None:
            return {
                "status": "error",
                "tool": "query_topology_neighbors",
                "project": project_name,
                "asset_query": asset_query,
                "selected_node_id": selected_node_id,
                "error": "selected_node_not_found",
                "answer": f"Der voraufgelöste Knoten wurde im Topologiegraph nicht gefunden: {selected_node_id}",
            }

        neighbor_payload = get_neighbor_payload(topology_graph, selected["node_id"])
        if neighbor_payload.get("status") != "ok":
            return {
                "status": "error",
                "tool": "query_topology_neighbors",
                "project": project_name,
                "asset_query": asset_query,
                "matches": matches or [],
                **neighbor_payload,
            }

        return {
            "status": "ok",
            "tool": "query_topology_neighbors",
            "project": project_name,
            "asset_query": asset_query,
            "matches": matches or [selected],
            **neighbor_payload,
        }

    resolved_matches = find_matching_nodes(
        graph=topology_graph,
        asset_query=asset_query,
        max_results=max_matches,
    )

    if not resolved_matches:
        sample_nodes = []
        for node_id, data in list(topology_graph.nodes(data=True))[:50]:
            sample_nodes.append({
                "name": data.get("name"),
                "pf_class": data.get("pf_class"),
                "kind": data.get("kind"),
                "full_name": data.get("full_name"),
            })

        return {
            "status": "error",
            "tool": "query_topology_neighbors",
            "project": project_name,
            "asset_query": asset_query,
            "error": "no_matching_asset",
            "answer": f"Kein passendes Asset im PowerFactory-Topologiegraph gefunden für: {asset_query}",
            "debug_match": {
                "graph_nodes": topology_graph.number_of_nodes(),
                "graph_edges": topology_graph.number_of_edges(),
                "sample_nodes": sample_nodes,
            },
        }

    selected = resolved_matches[0]
    neighbor_payload = get_neighbor_payload(topology_graph, selected["node_id"])

    if neighbor_payload.get("status") != "ok":
        return {
            "status": "error",
            "tool": "query_topology_neighbors",
            "project": project_name,
            "asset_query": asset_query,
            "matches": resolved_matches,
            **neighbor_payload,
        }

    return {
        "status": "ok",
        "tool": "query_topology_neighbors",
        "project": project_name,
        "asset_query": asset_query,
        "matches": resolved_matches,
        **neighbor_payload,
    }