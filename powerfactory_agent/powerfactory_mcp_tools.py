from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.pf_runner import _get_pf, _get_app, _activate_project_by_name

try:
    from cimpy.powerfactory_agent.agents.LLM_interpreterAgent import LLM_interpreterAgent
    from cimpy.powerfactory_agent.agents.PowerFactoryAgent import PowerFactoryAgent
    from cimpy.powerfactory_agent.agents.Result_interpreterAgent import Result_interpreterAgent
    from cimpy.powerfactory_agent.agents.LLM_resultAgent import LLM_resultAgent
except ImportError:
    from cimpy.powerfactory_agent.agents.LLM_interpreterAgent import LLM_interpreterAgent
    from cimpy.powerfactory_agent.agents.PowerFactoryAgent import PowerFactoryAgent
    from cimpy.powerfactory_agent.agents.Result_interpreterAgent import Result_interpreterAgent
    from cimpy.powerfactory_agent.agents.LLM_resultAgent import LLM_resultAgent

from cimpy.powerfactory_agent.powerfactory_topology_graph import (
    build_powerfactory_topology_graph_from_services,
    find_matching_nodes,
    find_matches_in_inventory,
    query_powerfactory_topology_neighbors_from_services,
)


def _kill_powerfactory_if_running() -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "powerfactory.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


# ------------------------------------------------------------------
# POWERFACTORY CONTEXT
# ------------------------------------------------------------------
def get_powerfactory_context(project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    pf = _get_pf()

    _kill_powerfactory_if_running()

    app = _get_app(pf)
    if app is None:
        return {
            "status": "error",
            "tool": "powerfactory_context",
            "error": "PowerFactory nicht erreichbar (GetApplication/GetApplicationExt ist None)",
        }

    ok = _activate_project_by_name(app, project_name)
    if not ok:
        return {
            "status": "error",
            "tool": "powerfactory_context",
            "error": f"Projekt konnte nicht aktiviert werden (nicht gefunden/kein Zugriff): {project_name}",
        }

    project = app.GetActiveProject()
    if project is None:
        return {
            "status": "error",
            "tool": "powerfactory_context",
            "error": "Projekt nicht aktiv (GetActiveProject() None)",
        }

    studycase = app.GetActiveStudyCase()
    if studycase is None:
        return {
            "status": "error",
            "tool": "powerfactory_context",
            "error": "Kein aktiver Study Case",
        }

    return {
        "status": "ok",
        "tool": "powerfactory_context",
        "app": app,
        "project": project,
        "studycase": studycase,
        "project_name": project_name,
    }


def build_powerfactory_services(project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    context = get_powerfactory_context(project_name=project_name)
    if context["status"] != "ok":
        return context

    project = context["project"]
    studycase = context["studycase"]

    interpreter = LLM_interpreterAgent(project, studycase)
    executor = PowerFactoryAgent(project, studycase)
    result_agent = Result_interpreterAgent()
    llm_result_agent = LLM_resultAgent()

    context.update({
        "interpreter": interpreter,
        "executor": executor,
        "result_agent": result_agent,
        "llm_result_agent": llm_result_agent,
    })
    return context


# ------------------------------------------------------------------
# INTERNAL HELPERS USING EXISTING SERVICES
# ------------------------------------------------------------------
def _get_load_catalog_from_services(services: Dict[str, Any]) -> Dict[str, Any]:
    interpreter = services["interpreter"]
    project_name = services["project_name"]

    if hasattr(interpreter, "get_load_catalog_metadata"):
        loads = interpreter.get_load_catalog_metadata()
    else:
        loads = []
        for entry in interpreter.catalog:
            loads.append({
                "loc_name": entry["loc_name"],
                "full_name": entry["full_name"],
                "tokens": sorted(entry["tokens"]),
            })

    return {
        "status": "ok",
        "tool": "get_load_catalog",
        "project": project_name,
        "loads": loads,
    }


def _interpret_instruction_with_services(services: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    interpreter = services["interpreter"]
    project_name = services["project_name"]

    instruction = interpreter.interpret(user_input)

    if isinstance(instruction, dict) and "error" in instruction:
        return {
            "status": "error",
            "tool": "interpret_instruction",
            "error": instruction.get("error"),
            "details": instruction.get("details"),
            "user_input": user_input,
            "project": project_name,
        }

    return {
        "status": "ok",
        "tool": "interpret_instruction",
        "user_input": user_input,
        "project": project_name,
        "instruction": instruction,
    }


def _resolve_load_with_services(services: Dict[str, Any], instruction: dict) -> Dict[str, Any]:
    interpreter = services["interpreter"]
    project_name = services["project_name"]

    try:
        if hasattr(interpreter, "resolve_with_metadata"):
            resolution = interpreter.resolve_with_metadata(instruction)
        else:
            pf_object = interpreter.resolve(instruction)
            resolution = {
                "status": "ok",
                "requested_load_name": instruction.get("load_name"),
                "selected": {
                    "loc_name": getattr(pf_object, "loc_name", None),
                    "full_name": pf_object.GetFullName() if hasattr(pf_object, "GetFullName") else None,
                },
            }

        return {
            "status": resolution.get("status", "ok"),
            "tool": "resolve_load",
            "project": project_name,
            "instruction": instruction,
            "resolution": resolution,
        }
    except Exception as e:
        return {
            "status": "error",
            "tool": "resolve_load",
            "project": project_name,
            "instruction": instruction,
            "error": "resolve_failed",
            "details": str(e),
        }


# ------------------------------------------------------------------
# GENERIC TOPOLOGY ENTITY HELPERS
# ------------------------------------------------------------------
def _safe_lower(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _tokenize(value: str) -> List[str]:
    text = _safe_lower(value)
    for ch in "\\/()[]{}:;,.!?\"'":
        text = text.replace(ch, " ")
    return [token for token in text.split() if token]


def _infer_entity_type_from_text(user_input: str, inventory: Dict[str, Any]) -> Optional[str]:
    text = _safe_lower(user_input)

    if "bus" in text or "knoten" in text or "terminal" in text:
        return "bus"
    if "last" in text or "load" in text:
        return "load"
    if "schalter" in text or "switch" in text or "breaker" in text or "coupler" in text:
        return "switch"
    if "trafo" in text or "transformer" in text:
        return "transformer"
    if "leitung" in text or "line" in text or "kabel" in text:
        return "line"
    if "generator" in text or "gen" in text:
        return "generator"

    counts_by_type = inventory.get("counts_by_type", {}) if isinstance(inventory, dict) else {}
    if len(counts_by_type) == 1:
        return next(iter(counts_by_type.keys()))

    return None


def _build_entity_name_candidates(user_input: str) -> List[str]:
    """
    No hard-coded filler removal.
    We only generate overlapping token windows from the full user input.
    The resolver later chooses the best candidate against the real inventory.
    """
    text = (user_input or "").strip()
    if not text:
        return []

    candidates: List[str] = []
    if text:
        candidates.append(text)

    tokens = _tokenize(text)
    for window_size in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - window_size + 1):
            candidate = " ".join(tokens[start:start + window_size]).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    return candidates


def _interpret_entity_instruction_with_services(
    services: Dict[str, Any],
    user_input: str,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]

    entity_type = _infer_entity_type_from_text(user_input, inventory)
    entity_name_candidates = _build_entity_name_candidates(user_input)

    instruction = {
        "query_type": "neighbors",
        "entity_type": entity_type,
        "entity_name_raw": user_input,
        "entity_name_candidates": entity_name_candidates,
        "available_types": inventory.get("available_types", []),
    }

    return {
        "status": "ok",
        "tool": "interpret_entity_instruction",
        "user_input": user_input,
        "project": project_name,
        "instruction": instruction,
    }


def _build_topology_inventory_with_services(
    services: Dict[str, Any],
    topology_graph_result: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]
    inventory = topology_graph_result.get("inventory", {}) if isinstance(topology_graph_result, dict) else {}

    return {
        "status": "ok",
        "tool": "build_topology_inventory",
        "project": project_name,
        "inventory": inventory,
    }


def _resolve_entity_from_inventory_with_services(
    services: Dict[str, Any],
    instruction: dict,
    inventory: Dict[str, Any],
    topology_graph: Any,
    max_matches: int = 10,
) -> Dict[str, Any]:
    project_name = services["project_name"]

    if topology_graph is None:
        return {
            "status": "error",
            "tool": "resolve_entity_from_inventory",
            "project": project_name,
            "instruction": instruction,
            "error": "missing_topology_graph",
            "details": "Es wurde kein Topologiegraph an die Entity-Auflösung übergeben.",
        }

    if not inventory:
        return {
            "status": "error",
            "tool": "resolve_entity_from_inventory",
            "project": project_name,
            "instruction": instruction,
            "error": "missing_inventory",
            "details": "Es wurde kein Topologie-Inventar übergeben.",
        }

    entity_type = instruction.get("entity_type")
    items_by_type = inventory.get("items_by_type", {}) if isinstance(inventory, dict) else {}

    if entity_type and entity_type in items_by_type:
        candidate_pool = items_by_type[entity_type]
        used_entity_type = entity_type
    else:
        candidate_pool = []
        used_entity_type = entity_type
        for _, items in items_by_type.items():
            candidate_pool.extend(items)

    if not candidate_pool:
        return {
            "status": "error",
            "tool": "resolve_entity_from_inventory",
            "project": project_name,
            "instruction": instruction,
            "error": "empty_candidate_pool",
            "details": "Für den gewünschten Entity-Typ wurden keine Kandidaten im Inventar gefunden.",
            "inventory_types": inventory.get("available_types", []),
        }

    raw_candidates = instruction.get("entity_name_candidates", []) if isinstance(instruction, dict) else []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    attempted_queries: List[Dict[str, Any]] = []
    selected_matches: List[Dict[str, Any]] = []
    selected_query: Optional[str] = None

    for query in raw_candidates:
        matches = find_matches_in_inventory(
            inventory_items=candidate_pool,
            raw_query=query,
            max_results=max_matches,
        )

        attempted_queries.append({
            "asset_query": query,
            "entity_type": used_entity_type,
            "match_count": len(matches),
            "top_matches": [
                {
                    "name": m.get("name"),
                    "pf_class": m.get("pf_class"),
                    "inventory_type": m.get("inventory_type"),
                    "score": m.get("score"),
                    "matched_query_variant": m.get("matched_query_variant"),
                }
                for m in matches[:5]
            ],
        })

        if matches:
            selected_query = query
            selected_matches = matches
            break

    if not selected_matches:
        sample_pool = [
            {
                "name": item.get("name"),
                "pf_class": item.get("pf_class"),
                "inventory_type": item.get("inventory_type"),
                "full_name": item.get("full_name"),
            }
            for item in candidate_pool[:50]
        ]
        return {
            "status": "error",
            "tool": "resolve_entity_from_inventory",
            "project": project_name,
            "instruction": instruction,
            "error": "no_matching_asset",
            "details": "Kein passendes Asset aus der Entity-Instruction konnte im Inventar gematcht werden.",
            "attempted_queries": attempted_queries,
            "debug_match": {
                "candidate_pool_size": len(candidate_pool),
                "entity_type": used_entity_type,
                "sample_candidates": sample_pool,
            },
        }

    selected_match = selected_matches[0]

    # Re-confirm selected node against graph for consistency/debug
    graph_matches = find_matching_nodes(
        graph=topology_graph,
        asset_query=selected_match.get("name") or selected_match.get("full_name") or "",
        max_results=max_matches,
        class_hint=used_entity_type,
    )

    return {
        "status": "ok",
        "tool": "resolve_entity_from_inventory",
        "project": project_name,
        "instruction": instruction,
        "asset_query": selected_match.get("name") or selected_match.get("full_name"),
        "selected_query": selected_query,
        "selected_match": selected_match,
        "matches": selected_matches,
        "graph_confirmation_matches": graph_matches,
        "attempted_queries": attempted_queries,
        "entity_type": used_entity_type,
    }


def _execute_change_load_with_services(services: Dict[str, Any], instruction: dict) -> Dict[str, Any]:
    app = services["app"]
    studycase = services["studycase"]
    interpreter = services["interpreter"]
    executor = services["executor"]
    project_name = services["project_name"]

    try:
        resolved_load = interpreter.resolve(instruction)
    except Exception as e:
        return {
            "status": "error",
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "error": "resolve_failed",
            "details": str(e),
        }

    try:
        ldf_list = list(studycase.GetContents("*.ComLdf", 1))
    except Exception:
        ldf_list = []

    if not ldf_list:
        ldf = studycase.CreateObject("ComLdf", "LoadFlow")
    else:
        ldf = ldf_list[0]

    ldf.Execute()

    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    u_before: Dict[str, float] = {}
    for bus in buses:
        u_before[bus.loc_name] = bus.GetAttribute("m:u")

    try:
        _ = resolved_load.GetAttribute("plini")
    except AttributeError:
        return {
            "status": "error",
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "error": f"Last {getattr(resolved_load, 'loc_name', '<unknown>')} hat kein Attribut 'plini'",
        }

    execution_result = executor.execute(instruction, resolved_load)

    ldf.Execute()

    u_after: Dict[str, float] = {}
    for bus in buses:
        u_after[bus.loc_name] = bus.GetAttribute("m:u")

    deltas: Dict[str, float] = {}
    for name, u0 in u_before.items():
        u1 = u_after.get(name)
        if u1 is not None:
            deltas[name] = u1 - u0

    return {
        "status": "ok",
        "tool": "execute_change_load",
        "project": project_name,
        "studycase": getattr(studycase, "loc_name", None),
        "instruction": instruction,
        "resolved_load": getattr(resolved_load, "loc_name", None),
        "execution": execution_result,
        "data": {
            "u_before": u_before,
            "u_after": u_after,
            "delta_u": deltas,
        },
    }


def _summarize_powerfactory_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    result_agent = services["result_agent"]
    llm_result_agent = services["llm_result_agent"]
    project_name = services["project_name"]

    data = result_payload.get("data", {}) if isinstance(result_payload, dict) else {}
    u_before = data.get("u_before", {}) if isinstance(data, dict) else {}
    u_after = data.get("u_after", {}) if isinstance(data, dict) else {}

    messages = result_agent.interpret_voltage_change(u_before, u_after)
    summary = llm_result_agent.summarize(messages, user_input)

    return {
        "status": "ok",
        "tool": "summarize_powerfactory_result",
        "project": project_name,
        "messages": messages,
        "answer": summary,
    }


# ------------------------------------------------------------------
# PUBLIC TOOL FUNCTIONS
# ------------------------------------------------------------------
def get_load_catalog(project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services
    return _get_load_catalog_from_services(services)


def interpret_instruction(user_input: str, project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services
    return _interpret_instruction_with_services(services, user_input)


def resolve_load(instruction: dict, project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services
    return _resolve_load_with_services(services, instruction)


def execute_change_load(instruction: dict, project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services
    return _execute_change_load_with_services(services, instruction)


def summarize_powerfactory_result(
    result_payload: dict,
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services
    return _summarize_powerfactory_result_with_services(
        services=services,
        result_payload=result_payload,
        user_input=user_input,
    )


def interpret_entity_instruction(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
    contract_cubicles: bool = True,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    graph_result = build_powerfactory_topology_graph_from_services(
        services=services,
        contract_cubicles=contract_cubicles,
    )
    if graph_result["status"] != "ok":
        return graph_result

    return _interpret_entity_instruction_with_services(
        services=services,
        user_input=user_input,
        inventory=graph_result.get("inventory", {}),
    )


def resolve_entity_from_inventory(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
    contract_cubicles: bool = True,
    max_matches: int = 10,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    graph_result = build_powerfactory_topology_graph_from_services(
        services=services,
        contract_cubicles=contract_cubicles,
    )
    if graph_result["status"] != "ok":
        return graph_result

    return _resolve_entity_from_inventory_with_services(
        services=services,
        instruction=instruction,
        inventory=graph_result.get("inventory", {}),
        topology_graph=graph_result.get("topology_graph"),
        max_matches=max_matches,
    )


# ------------------------------------------------------------------
# CONVENIENCE TOOLS
# ------------------------------------------------------------------
def run_powerfactory_pipeline(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    interpretation = _interpret_instruction_with_services(
        services=services,
        user_input=user_input,
    )
    if interpretation["status"] != "ok":
        return interpretation

    instruction = interpretation["instruction"]

    resolution = _resolve_load_with_services(
        services=services,
        instruction=instruction,
    )
    if resolution["status"] != "ok":
        return resolution

    execution = _execute_change_load_with_services(
        services=services,
        instruction=instruction,
    )
    if execution["status"] != "ok":
        return execution

    summary = _summarize_powerfactory_result_with_services(
        services=services,
        result_payload=execution,
        user_input=user_input,
    )
    if summary["status"] != "ok":
        return summary

    return {
        "status": "ok",
        "tool": "powerfactory",
        "project": project_name,
        "studycase": execution.get("studycase"),
        "instruction": instruction,
        "resolved_load": execution.get("resolved_load"),
        "data": execution.get("data", {}),
        "messages": summary.get("messages", []),
        "answer": summary.get("answer", ""),
        "debug": {
            "interpretation": interpretation,
            "resolution": resolution,
            "execution": execution,
            "summary": summary,
        },
    }


def run_powerfactory_topology_pipeline(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
    contract_cubicles: bool = True,
    max_matches: int = 10,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    graph_result = build_powerfactory_topology_graph_from_services(
        services=services,
        contract_cubicles=contract_cubicles,
    )
    if graph_result["status"] != "ok":
        return graph_result

    inventory_result = _build_topology_inventory_with_services(
        services=services,
        topology_graph_result=graph_result,
    )
    if inventory_result["status"] != "ok":
        return inventory_result

    interpretation = _interpret_entity_instruction_with_services(
        services=services,
        user_input=user_input,
        inventory=inventory_result["inventory"],
    )
    if interpretation["status"] != "ok":
        return interpretation

    resolution = _resolve_entity_from_inventory_with_services(
        services=services,
        instruction=interpretation["instruction"],
        inventory=inventory_result["inventory"],
        topology_graph=graph_result["topology_graph"],
        max_matches=max_matches,
    )
    if resolution["status"] != "ok":
        return resolution

    result = query_powerfactory_topology_neighbors_from_services(
        services=services,
        topology_graph=graph_result["topology_graph"],
        asset_query=resolution.get("asset_query", user_input),
        selected_node_id=(resolution.get("selected_match") or {}).get("node_id"),
        matches=resolution.get("matches", []),
        max_matches=max_matches,
    )
    if result["status"] != "ok":
        return result

    return {
        "status": "ok",
        "tool": "powerfactory_topology",
        "project": project_name,
        "answer": result.get("answer", ""),
        "graph_result": graph_result,
        "inventory_result": inventory_result,
        "interpretation": interpretation,
        "resolution": resolution,
        "topology_result": result,
    }