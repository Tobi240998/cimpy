from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.langchain_llm import get_llm
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


# ------------------------------------------------------------------
# LLM OUTPUT MODEL FOR SWITCH MATCHING
# ------------------------------------------------------------------
class SwitchMatchDecision(BaseModel):
    selected_switch_name: Optional[str] = Field(
        default=None,
        description="Exact switch name from the provided candidate list, or null if no safe match exists."
    )
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    alternatives: List[str] = Field(default_factory=list)
    should_execute: bool = Field(
        description="True only if the selected switch is a safe unambiguous choice."
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


def _to_py_list(value: Any) -> List[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


# ------------------------------------------------------------------
# POWERFACTORY CONTEXT
# ------------------------------------------------------------------
def get_powerfactory_context(project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    pf = _get_pf()
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
# LOAD CATALOG / LOAD INTERPRETATION
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

    instruction = _ensure_instruction_result_requests(instruction, user_input=user_input)

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
# GENERIC TEXT / INVENTORY HELPERS
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


def _build_entity_name_candidates(user_input: str) -> List[str]:
    text = (user_input or "").strip()
    if not text:
        return []

    candidates: List[str] = []
    candidates.append(text)

    tokens = _tokenize(text)
    for window_size in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - window_size + 1):
            candidate = " ".join(tokens[start:start + window_size]).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    return candidates


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


# ------------------------------------------------------------------
# TOPOLOGY INVENTORY / TOPOLOGY INTERPRETATION
# ------------------------------------------------------------------
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


def _interpret_entity_instruction_with_services(
    services: Dict[str, Any],
    user_input: str,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]

    instruction = {
        "query_type": "neighbors",
        "entity_type": _infer_entity_type_from_text(user_input, inventory),
        "entity_name_raw": user_input,
        "entity_name_candidates": _build_entity_name_candidates(user_input),
        "available_types": inventory.get("available_types", []),
    }

    return {
        "status": "ok",
        "tool": "interpret_entity_instruction",
        "user_input": user_input,
        "project": project_name,
        "instruction": instruction,
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
        return {
            "status": "error",
            "tool": "resolve_entity_from_inventory",
            "project": project_name,
            "instruction": instruction,
            "error": "no_matching_asset",
            "details": "Kein passendes Asset aus der Entity-Instruction konnte im Inventar gematcht werden.",
            "attempted_queries": attempted_queries,
        }

    selected_match = selected_matches[0]

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


# ------------------------------------------------------------------
# SWITCH INVENTORY / SWITCH INTERPRETATION / SWITCH RESOLUTION
# ------------------------------------------------------------------
def _looks_like_switch_object(obj: Any) -> bool:
    try:
        cls = (obj.GetClassName() or "").lower()
    except Exception:
        cls = ""

    try:
        name = (obj.loc_name or "").lower()
    except Exception:
        name = ""

    if "switch" in cls or "coup" in cls or cls in {"staswit", "elmcoup", "relfuse"}:
        return True

    if "switch" in name or "schalter" in name or "breaker" in name or "coupler" in name:
        return True

    return False


def _build_switch_inventory_from_services(services: Dict[str, Any]) -> Dict[str, Any]:
    app = services["app"]
    project_name = services["project_name"]

    switches: List[Dict[str, Any]] = []

    sta_objects = app.GetCalcRelevantObjects("*.Sta*") or []
    for obj in sta_objects:
        if not _looks_like_switch_object(obj):
            continue

        try:
            full_name = obj.GetFullName()
        except Exception:
            full_name = None

        try:
            pf_class = obj.GetClassName()
        except Exception:
            pf_class = None

        try:
            name = obj.loc_name
        except Exception:
            name = None

        switches.append({
            "node_id": full_name or name,
            "name": name,
            "pf_class": pf_class,
            "full_name": full_name,
            "kind": "sta",
            "degree": 0,
            "inventory_type": "switch",
        })

    switches.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("full_name") or "")))

    return {
        "status": "ok",
        "tool": "build_switch_inventory",
        "project": project_name,
        "switches": switches,
    }


def _build_switch_inventory_payload(switches: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "available_types": ["switch"] if switches else [],
        "counts_by_type": {"switch": len(switches)},
        "items_by_type": {"switch": switches},
        "samples_by_type": {
            "switch": [
                {
                    "name": item.get("name"),
                    "pf_class": item.get("pf_class"),
                    "full_name": item.get("full_name"),
                }
                for item in switches[:10]
            ]
        },
    }


def _interpret_switch_instruction_with_services(
    services: Dict[str, Any],
    user_input: str,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]
    text = _safe_lower(user_input)

    operation = None
    if any(token in text for token in ["öffne", "oeffne", "open", "trenne"]):
        operation = "open"
    elif any(token in text for token in ["schließe", "schliesse", "schliese", "close", "einschalten", "zuschalten"]):
        operation = "close"
    elif any(token in text for token in ["toggle", "umschalten"]):
        operation = "toggle"

    instruction = {
        "query_type": "switch_operation",
        "operation": operation,
        "entity_type": "switch",
        "entity_name_raw": user_input,
        "entity_name_candidates": _build_entity_name_candidates(user_input),
        "available_types": inventory.get("available_types", []),
    }

    if not operation:
        return {
            "status": "error",
            "tool": "interpret_switch_instruction",
            "project": project_name,
            "user_input": user_input,
            "error": "missing_switch_operation",
            "details": "Es konnte keine Schalteroperation erkannt werden.",
            "instruction": instruction,
        }

    return {
        "status": "ok",
        "tool": "interpret_switch_instruction",
        "user_input": user_input,
        "project": project_name,
        "instruction": instruction,
    }


def _build_switch_match_chain():
    parser = PydanticOutputParser(pydantic_object=SwitchMatchDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You match a user's requested switch to an existing PowerFactory switch.\n"
            "You may only select a switch name that appears exactly in the provided candidate list.\n"
            "Do not invent names.\n"
            "If there is no safe unambiguous match, return selected_switch_name=null and should_execute=false.\n"
            "Use high confidence only for a clearly dominant match.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Available switch candidates:\n{switch_candidates}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser


def _resolve_switch_from_inventory_llm_with_services(
    services: Dict[str, Any],
    instruction: dict,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]

    items_by_type = inventory.get("items_by_type", {}) if isinstance(inventory, dict) else {}
    switch_candidates = items_by_type.get("switch", []) or []

    if not switch_candidates:
        return {
            "status": "error",
            "tool": "resolve_switch_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": "no_switch_candidates",
            "details": "Im Switch-Inventar wurden keine Schalterkandidaten gefunden.",
        }

    candidate_names = [item.get("name") for item in switch_candidates if item.get("name")]
    if not candidate_names:
        return {
            "status": "error",
            "tool": "resolve_switch_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": "empty_switch_names",
            "details": "Die vorhandenen Switch-Kandidaten haben keine verwertbaren Namen.",
        }

    try:
        chain, parser = _build_switch_match_chain()
        decision = chain.invoke({
            "user_input": instruction.get("entity_name_raw") or "",
            "switch_candidates": "\n".join(f"- {name}" for name in candidate_names),
            "format_instructions": parser.get_format_instructions(),
        })
    except Exception as e:
        return {
            "status": "error",
            "tool": "resolve_switch_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": "llm_switch_match_failed",
            "details": str(e),
        }

    selected_name = decision.selected_switch_name
    if selected_name not in candidate_names:
        return {
            "status": "error",
            "tool": "resolve_switch_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": "invalid_switch_selection",
            "details": "Das LLM hat keinen gültigen exakten Switch-Namen aus der Kandidatenliste zurückgegeben.",
            "llm_decision": decision.model_dump(),
            "candidate_names": candidate_names,
        }

    if not decision.should_execute or decision.confidence.lower() != "high":
        return {
            "status": "error",
            "tool": "resolve_switch_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": "switch_match_not_safe",
            "details": "Das LLM hat keinen ausreichend sicheren Switch-Treffer gefunden.",
            "llm_decision": decision.model_dump(),
            "candidate_names": candidate_names,
        }

    selected_match = next(item for item in switch_candidates if item.get("name") == selected_name)

    return {
        "status": "ok",
        "tool": "resolve_switch_from_inventory_llm",
        "project": project_name,
        "instruction": instruction,
        "asset_query": selected_name,
        "selected_match": selected_match,
        "matches": [selected_match],
        "llm_decision": decision.model_dump(),
    }


# ------------------------------------------------------------------
# SWITCH EXECUTION
# ------------------------------------------------------------------
def _get_object_by_full_name(app: Any, full_name: str) -> Any | None:
    if not full_name:
        return None

    try:
        obj = app.GetObject(full_name)
        if obj is not None:
            return obj
    except Exception:
        pass

    try:
        all_sta = app.GetCalcRelevantObjects("*.Sta*") or []
        for obj in all_sta:
            try:
                if obj.GetFullName() == full_name:
                    return obj
            except Exception:
                pass
    except Exception:
        pass

    try:
        all_elm = app.GetCalcRelevantObjects("*.Elm*") or []
        for obj in all_elm:
            try:
                if obj.GetFullName() == full_name:
                    return obj
            except Exception:
                pass
    except Exception:
        pass

    return None


def _read_switch_state(obj: Any) -> Dict[str, Any]:
    candidates = ["on_off", "isclosed", "closed", "outserv"]

    for attr_name in candidates:
        try:
            value = obj.GetAttribute(attr_name)
            if value is not None:
                return {"status": "ok", "state_source": attr_name, "raw_value": value}
        except Exception:
            pass

        try:
            value = getattr(obj, attr_name)
            if value is not None:
                return {"status": "ok", "state_source": attr_name, "raw_value": value}
        except Exception:
            pass

    return {
        "status": "error",
        "error": "switch_state_not_readable",
        "details": "Kein bekannter Zustandsindikator gefunden.",
    }


def _normalize_switch_state(raw_value: Any, source: str) -> Optional[str]:
    if raw_value is None:
        return None

    source = (source or "").lower()

    try:
        ivalue = int(raw_value)
    except Exception:
        ivalue = None

    if source == "outserv":
        if ivalue == 0:
            return "closed"
        if ivalue == 1:
            return "open"

    if source in {"on_off", "isclosed", "closed"}:
        if ivalue == 1:
            return "closed"
        if ivalue == 0:
            return "open"

    text = str(raw_value).strip().lower()
    if text in {"1", "true", "closed", "on"}:
        return "closed"
    if text in {"0", "false", "open", "off"}:
        return "open"

    return None


def _set_attr_or_field(obj: Any, attr_name: str, value: Any) -> bool:
    try:
        obj.SetAttribute(attr_name, value)
        return True
    except Exception:
        pass

    try:
        setattr(obj, attr_name, value)
        return True
    except Exception:
        pass

    return False


def _apply_switch_state_to_object(obj: Any, operation: str) -> Dict[str, Any]:
    op = (operation or "").lower()

    method_map = {
        "open": ["Open", "SwitchOff", "OpenBreaker", "OpenSwitch"],
        "close": ["Close", "SwitchOn", "CloseBreaker", "CloseSwitch"],
    }

    if op in method_map:
        for method_name in method_map[op]:
            try:
                method = getattr(obj, method_name)
                result = method()
                return {
                    "status": "ok",
                    "apply_mode": "method",
                    "apply_target": method_name,
                    "method_result": result,
                }
            except Exception:
                pass

    attr_attempts = []
    if op == "open":
        attr_attempts = [("on_off", 0), ("isclosed", 0), ("closed", 0), ("outserv", 1)]
    elif op == "close":
        attr_attempts = [("on_off", 1), ("isclosed", 1), ("closed", 1), ("outserv", 0)]

    for attr_name, value in attr_attempts:
        if _set_attr_or_field(obj, attr_name, value):
            return {
                "status": "ok",
                "apply_mode": "attribute",
                "apply_target": attr_name,
                "applied_value": value,
            }

    if op == "toggle":
        before_state_raw = _read_switch_state(obj)
        if before_state_raw.get("status") == "ok":
            state = _normalize_switch_state(
                raw_value=before_state_raw.get("raw_value"),
                source=before_state_raw.get("state_source", ""),
            )
            if state == "open":
                return _apply_switch_state_to_object(obj, "close")
            if state == "closed":
                return _apply_switch_state_to_object(obj, "open")

    return {
        "status": "error",
        "error": "switch_operation_not_supported",
        "details": "Für dieses Schalterobjekt konnte keine unterstützte Methode oder kein unterstütztes Attribut gefunden werden.",
    }


def _execute_switch_operation_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
    run_loadflow_after: bool = True,
) -> Dict[str, Any]:
    app = services["app"]
    studycase = services["studycase"]
    project_name = services["project_name"]

    selected_match = resolution.get("selected_match") if isinstance(resolution, dict) else None
    if not isinstance(selected_match, dict):
        return {
            "status": "error",
            "tool": "execute_switch_operation",
            "project": project_name,
            "instruction": instruction,
            "error": "missing_selected_switch",
            "details": "Es wurde kein aufgelöstes Schalterobjekt übergeben.",
        }

    operation = instruction.get("operation")
    if not operation:
        return {
            "status": "error",
            "tool": "execute_switch_operation",
            "project": project_name,
            "instruction": instruction,
            "error": "missing_switch_operation",
            "details": "In der Instruction fehlt die gewünschte Schalteroperation.",
        }

    full_name = selected_match.get("full_name")
    switch_obj = _get_object_by_full_name(app, full_name)
    if switch_obj is None:
        return {
            "status": "error",
            "tool": "execute_switch_operation",
            "project": project_name,
            "instruction": instruction,
            "resolution": resolution,
            "error": "switch_object_not_found",
            "details": f"Das aufgelöste Objekt konnte in PowerFactory nicht geladen werden: {full_name}",
        }

    before_state_info = _read_switch_state(switch_obj)
    before_state = None
    if before_state_info.get("status") == "ok":
        before_state = _normalize_switch_state(
            raw_value=before_state_info.get("raw_value"),
            source=before_state_info.get("state_source", ""),
        )

    apply_result = _apply_switch_state_to_object(switch_obj, operation)
    if apply_result.get("status") != "ok":
        return {
            "status": "error",
            "tool": "execute_switch_operation",
            "project": project_name,
            "instruction": instruction,
            "resolution": resolution,
            "switch": {
                "name": getattr(switch_obj, "loc_name", None),
                "full_name": full_name,
                "pf_class": switch_obj.GetClassName() if hasattr(switch_obj, "GetClassName") else None,
            },
            **apply_result,
        }

    loadflow_info = {"executed": False, "loadflow_command": None}
    if run_loadflow_after:
        try:
            ldf_list = _to_py_list(studycase.GetContents("*.ComLdf", 1))
            if not ldf_list:
                ldf = studycase.CreateObject("ComLdf", "LoadFlow")
            else:
                ldf = ldf_list[0]

            ldf.Execute()
            loadflow_info = {
                "executed": True,
                "loadflow_command": getattr(ldf, "loc_name", "LoadFlow"),
            }
        except Exception as e:
            loadflow_info = {"executed": False, "loadflow_command": None, "error": str(e)}

    after_state_info = _read_switch_state(switch_obj)
    after_state = None
    if after_state_info.get("status") == "ok":
        after_state = _normalize_switch_state(
            raw_value=after_state_info.get("raw_value"),
            source=after_state_info.get("state_source", ""),
        )

    return {
        "status": "ok",
        "tool": "execute_switch_operation",
        "project": project_name,
        "studycase": getattr(studycase, "loc_name", None),
        "instruction": instruction,
        "resolution": resolution,
        "switch": {
            "name": getattr(switch_obj, "loc_name", None),
            "full_name": full_name,
            "pf_class": switch_obj.GetClassName() if hasattr(switch_obj, "GetClassName") else None,
        },
        "state_before": before_state,
        "state_after": after_state,
        "state_before_raw": before_state_info,
        "state_after_raw": after_state_info,
        "apply_result": apply_result,
        "loadflow": loadflow_info,
    }


def _summarize_switch_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    project_name = services["project_name"]

    switch = result_payload.get("switch", {}) if isinstance(result_payload, dict) else {}
    instruction = result_payload.get("instruction", {}) if isinstance(result_payload, dict) else {}

    switch_name = switch.get("name") or switch.get("full_name") or "<unbekannt>"
    pf_class = switch.get("pf_class") or "<unknown>"
    operation = instruction.get("operation") or "<unknown>"
    state_before = result_payload.get("state_before")
    state_after = result_payload.get("state_after")
    loadflow = result_payload.get("loadflow", {}) if isinstance(result_payload, dict) else {}

    if state_before and state_after:
        answer = (
            f"Die Schalteroperation '{operation}' wurde für '{switch_name}' ({pf_class}) ausgeführt. "
            f"Zustand vorher: {state_before}. Zustand nachher: {state_after}."
        )
    else:
        answer = f"Die Schalteroperation '{operation}' wurde für '{switch_name}' ({pf_class}) ausgeführt."

    if loadflow.get("executed"):
        answer += " Anschließend wurde ein Lastfluss gerechnet."

    return {
        "status": "ok",
        "tool": "summarize_switch_result",
        "project": project_name,
        "answer": answer,
        "messages": [],
    }



# ------------------------------------------------------------------
# GENERIC LOAD RESULT SNAPSHOTS / INTERPRETATION
# ------------------------------------------------------------------
METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "bus_voltage": {
        "label": "Bus-Spannung",
        "unit": "p.u.",
        "legacy_before_key": "u_before",
        "legacy_after_key": "u_after",
        "legacy_delta_key": "delta_u",
        "aliases": [
            "spannung", "spannungen", "busspannung", "busspannungen", "voltage", "voltages",
            "u", "u_bus", "u_knoten",
        ],
    },
    "bus_p": {
        "label": "Knoten-Wirkleistung",
        "unit": "MW",
        "aliases": [
            "wirkleistung", "wirkleistungen", "knotenwirkleistung", "knotenwirkleistungen",
            "bus p", "knoten p", "active power", "bus_p", "p",
        ],
    },
    "bus_q": {
        "label": "Knoten-Blindleistung",
        "unit": "MVAr",
        "aliases": [
            "blindleistung", "blindleistungen", "knotenblindleistung", "knotenblindleistungen",
            "bus q", "knoten q", "reactive power", "bus_q", "q",
        ],
    },
    "line_loading": {
        "label": "Leitungsauslastung",
        "unit": "%",
        "aliases": [
            "auslastung", "leitungsauslastung", "leitungsauslastungen", "leitungsauslastung",
            "leitungsauslastungen", "line loading", "loading", "line_loading", "line loading",
        ],
    },
}

DEFAULT_RESULT_REQUESTS: List[str] = ["bus_voltage"]


def _infer_result_requests_from_user_input(user_input: str) -> List[str]:
    text = _safe_lower(user_input)
    if not text:
        return list(DEFAULT_RESULT_REQUESTS)

    inferred: List[str] = []
    for metric_name, spec in METRIC_SPECS.items():
        aliases = spec.get("aliases", []) or []
        if any(alias in text for alias in aliases):
            inferred.append(metric_name)

    if inferred:
        return inferred

    return list(DEFAULT_RESULT_REQUESTS)



def _normalize_result_requests(requested_metrics: Any, user_input: str = "") -> List[str]:
    normalized: List[str] = []

    if isinstance(requested_metrics, str):
        requested_metrics = [requested_metrics]
    elif not isinstance(requested_metrics, list):
        requested_metrics = []

    alias_to_metric: Dict[str, str] = {}
    for metric_name, spec in METRIC_SPECS.items():
        alias_to_metric[metric_name] = metric_name
        for alias in spec.get("aliases", []) or []:
            alias_to_metric[_safe_lower(alias)] = metric_name

    for raw_metric in requested_metrics:
        key = _safe_lower(raw_metric)
        if not key:
            continue
        metric = alias_to_metric.get(key)
        if metric and metric not in normalized:
            normalized.append(metric)

    if normalized:
        return normalized

    return _infer_result_requests_from_user_input(user_input)



def _ensure_instruction_result_requests(instruction: Any, user_input: str = "") -> Dict[str, Any]:
    if isinstance(instruction, BaseModel):
        try:
            instruction = instruction.model_dump()
        except Exception:
            try:
                instruction = instruction.dict()
            except Exception:
                instruction = {}
    elif not isinstance(instruction, dict):
        instruction = {}
    else:
        instruction = dict(instruction)

    instruction["result_requests"] = _normalize_result_requests(
        instruction.get("result_requests", []),
        user_input=user_input,
    )
    return instruction


def _safe_get_pf_attribute(obj: Any, attr_name: str) -> Any:
    try:
        return obj.GetAttribute(attr_name)
    except Exception:
        return None


def _build_pf_object_identity(obj: Any) -> Dict[str, Any]:
    name = None
    pf_class = None
    full_name = None

    try:
        name = getattr(obj, "loc_name", None)
    except Exception:
        pass

    try:
        pf_class = obj.GetClassName()
    except Exception:
        pass

    try:
        full_name = obj.GetFullName()
    except Exception:
        pass

    return {
        "name": name,
        "pf_class": pf_class,
        "full_name": full_name,
    }


def _coerce_numeric_pf_value(value: Any) -> Tuple[bool, float | None]:
    if value is None:
        return False, None

    if isinstance(value, bool):
        return True, float(value)

    if isinstance(value, (int, float)):
        return True, float(value)

    try:
        return True, float(value)
    except Exception:
        pass

    try:
        if hasattr(value, "__len__") and len(value) == 1:
            return True, float(value[0])
    except Exception:
        pass

    return False, None



def _read_first_available_attribute_with_debug(
    obj: Any,
    attr_candidates: List[str],
    identity: Dict[str, Any],
) -> Dict[str, Any]:
    non_numeric_hits: List[Dict[str, Any]] = []

    for attr_name in attr_candidates:
        value = _safe_get_pf_attribute(obj, attr_name)
        if value is None:
            continue

        is_numeric, numeric_value = _coerce_numeric_pf_value(value)
        if is_numeric:
            return {
                "ok": True,
                "value": numeric_value,
                "raw_value": value,
                "used_attr": attr_name,
                "tried_attrs": attr_candidates,
                "non_numeric_hits": non_numeric_hits,
                **identity,
            }

        non_numeric_hits.append({
            "attr": attr_name,
            "value_repr": repr(value),
            "value_type": type(value).__name__,
        })

    return {
        "ok": False,
        "value": None,
        "used_attr": None,
        "tried_attrs": attr_candidates,
        "non_numeric_hits": non_numeric_hits,
        **identity,
    }



def _read_bus_voltage_pu_with_debug(bus: Any) -> Dict[str, Any]:
    identity = _build_pf_object_identity(bus)
    identity["bus_name"] = identity.get("name")
    tried_attrs = ["m:u", "m:ul", "m:Ul", "m:U"]
    return _read_first_available_attribute_with_debug(bus, tried_attrs, identity)



def _read_bus_p_with_debug(bus: Any) -> Dict[str, Any]:
    identity = _build_pf_object_identity(bus)
    identity["bus_name"] = identity.get("name")
    tried_attrs = [
        "m:Psum:bus1",
        "m:Psum:bus2",
        "m:Psum",
        "c:Psum",
        "m:P:bus1",
        "m:P:bus2",
        "m:Pbus1",
        "m:Pbus2",
        "m:P1",
        "m:P2",
        "m:P",
        "c:P",
    ]
    return _read_first_available_attribute_with_debug(bus, tried_attrs, identity)



def _read_bus_q_with_debug(bus: Any) -> Dict[str, Any]:
    identity = _build_pf_object_identity(bus)
    identity["bus_name"] = identity.get("name")
    tried_attrs = [
        "m:Qsum:bus1",
        "m:Qsum:bus2",
        "m:Qsum",
        "c:Qsum",
        "m:Q:bus1",
        "m:Q:bus2",
        "m:Qbus1",
        "m:Qbus2",
        "m:Q1",
        "m:Q2",
        "m:Q",
        "c:Q",
    ]
    return _read_first_available_attribute_with_debug(bus, tried_attrs, identity)



def _read_line_loading_with_debug(line: Any) -> Dict[str, Any]:
    identity = _build_pf_object_identity(line)
    identity["line_name"] = identity.get("name")
    tried_attrs = [
        "c:loading",
        "m:loading",
        "c:loading1",
        "m:loading1",
        "c:Loading",
        "m:Loading",
        "c:load",
        "m:load",
        "c:Load",
        "m:Load",
    ]
    return _read_first_available_attribute_with_debug(line, tried_attrs, identity)


def _snapshot_objects_with_debug(
    app: Any,
    object_queries: List[str],
    reader_fn: Any,
    object_label: str,
    identity_name_key: str,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    missing: List[Dict[str, Any]] = []
    attr_usage: Dict[str, int] = {}
    total_objects = 0

    seen_ids: set[str] = set()

    for query in object_queries:
        objects = app.GetCalcRelevantObjects(query) or []
        total_objects += len(objects)

        for obj in objects:
            identity = _build_pf_object_identity(obj)
            object_id = identity.get("full_name") or identity.get("name")
            if not object_id or object_id in seen_ids:
                continue
            seen_ids.add(object_id)

            read_result = reader_fn(obj)
            object_name = (
                read_result.get(identity_name_key)
                or read_result.get("name")
                or read_result.get("full_name")
            )
            if not object_name:
                continue

            if read_result["ok"]:
                snapshot[object_name] = read_result["value"]
                used_attr = read_result.get("used_attr")
                if used_attr:
                    attr_usage[used_attr] = attr_usage.get(used_attr, 0) + 1
            else:
                missing.append({
                    object_label: object_name,
                    "pf_class": read_result.get("pf_class"),
                    "full_name": read_result.get("full_name"),
                    "tried_attrs": read_result.get("tried_attrs", []),
                    "non_numeric_hits": read_result.get("non_numeric_hits", []),
                })

    return {
        "values": snapshot,
        "debug": {
            f"num_{object_label}_total": total_objects,
            f"num_{object_label}_with_value": len(snapshot),
            f"num_{object_label}_missing_value": len(missing),
            f"missing_{object_label}": missing,
            "attribute_usage": attr_usage,
            "object_queries": object_queries,
        },
    }


def _snapshot_bus_voltages_with_debug(app: Any) -> Dict[str, Any]:
    result = _snapshot_objects_with_debug(
        app=app,
        object_queries=["*.ElmTerm"],
        reader_fn=_read_bus_voltage_pu_with_debug,
        object_label="buses",
        identity_name_key="bus_name",
    )
    return {
        "voltages": result["values"],
        "debug": result["debug"],
    }


def _snapshot_bus_p_with_debug(app: Any) -> Dict[str, Any]:
    return _snapshot_objects_with_debug(
        app=app,
        object_queries=["*.ElmTerm"],
        reader_fn=_read_bus_p_with_debug,
        object_label="buses",
        identity_name_key="bus_name",
    )


def _snapshot_bus_q_with_debug(app: Any) -> Dict[str, Any]:
    return _snapshot_objects_with_debug(
        app=app,
        object_queries=["*.ElmTerm"],
        reader_fn=_read_bus_q_with_debug,
        object_label="buses",
        identity_name_key="bus_name",
    )


def _snapshot_line_loading_with_debug(app: Any) -> Dict[str, Any]:
    return _snapshot_objects_with_debug(
        app=app,
        object_queries=["*.ElmLne", "*.ElmCabl"],
        reader_fn=_read_line_loading_with_debug,
        object_label="lines",
        identity_name_key="line_name",
    )


def _collect_requested_metric_snapshots(app: Any, result_requests: List[str]) -> Dict[str, Any]:
    before_data: Dict[str, Dict[str, Any]] = {}
    snapshot_debug: Dict[str, Dict[str, Any]] = {}

    for metric in result_requests:
        if metric == "bus_voltage":
            snapshot_result = _snapshot_bus_voltages_with_debug(app)
            before_data[metric] = snapshot_result.get("voltages", {})
            snapshot_debug[metric] = snapshot_result.get("debug", {})
        elif metric == "bus_p":
            snapshot_result = _snapshot_bus_p_with_debug(app)
            before_data[metric] = snapshot_result.get("values", {})
            snapshot_debug[metric] = snapshot_result.get("debug", {})
        elif metric == "bus_q":
            snapshot_result = _snapshot_bus_q_with_debug(app)
            before_data[metric] = snapshot_result.get("values", {})
            snapshot_debug[metric] = snapshot_result.get("debug", {})
        elif metric == "line_loading":
            snapshot_result = _snapshot_line_loading_with_debug(app)
            before_data[metric] = snapshot_result.get("values", {})
            snapshot_debug[metric] = snapshot_result.get("debug", {})

    return {
        "values": before_data,
        "debug": snapshot_debug,
    }


def _compute_numeric_delta(before_map: Dict[str, Any], after_map: Dict[str, Any]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    common_names = set(before_map.keys()) & set(after_map.keys())

    for name in sorted(common_names):
        try:
            deltas[name] = float(after_map[name]) - float(before_map[name])
        except Exception:
            continue

    return deltas


def _build_metric_delta_payload(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    requested_metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    return {
        metric: _compute_numeric_delta(
            before_map=before.get(metric, {}),
            after_map=after.get(metric, {}),
        )
        for metric in requested_metrics
    }


def _build_metric_metadata(requested_metrics: List[str]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for metric in requested_metrics:
        spec = METRIC_SPECS.get(metric, {})
        metadata[metric] = {
            "label": spec.get("label", metric),
            "unit": spec.get("unit"),
        }
    return metadata


def _build_top_delta_lines(
    metric_label: str,
    delta_map: Dict[str, float],
    unit: str,
    top_n: int = 5,
) -> List[str]:
    if not delta_map:
        return [f"Für {metric_label} konnten keine vergleichbaren Vorher/Nachher-Werte gebildet werden."]

    sorted_items = sorted(
        delta_map.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    top_items = sorted_items[:top_n]
    parts = [
        f"{name}: {value:+.4f} {unit}".rstrip()
        for name, value in top_items
    ]

    return [
        f"Größte Änderungen für {metric_label}: " + ", ".join(parts),
    ]


def _build_metric_messages(
    metric: str,
    before_map: Dict[str, Any],
    after_map: Dict[str, Any],
    delta_map: Dict[str, float],
    result_agent: Any,
) -> List[str]:
    spec = METRIC_SPECS.get(metric, {})
    metric_label = spec.get("label", metric)
    unit = spec.get("unit", "")

    if metric == "bus_voltage" and hasattr(result_agent, "interpret_voltage_change"):
        try:
            return result_agent.interpret_voltage_change(before_map, after_map)
        except Exception:
            pass

    messages: List[str] = []
    messages.append(
        f"Für {metric_label} wurden {len(before_map)} Vorher-Werte, {len(after_map)} Nachher-Werte "
        f"und {len(delta_map)} Differenzen ermittelt."
    )
    messages.extend(_build_top_delta_lines(metric_label=metric_label, delta_map=delta_map, unit=unit))
    return messages


def _extract_metric_payload_from_result_payload(result_payload: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    data = result_payload.get("data", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(data, dict):
        return [], {}, {}, {}

    requested_metrics = data.get("requested_metrics")
    before = data.get("before")
    after = data.get("after")
    delta = data.get("delta")

    if isinstance(requested_metrics, list) and isinstance(before, dict) and isinstance(after, dict) and isinstance(delta, dict):
        return requested_metrics, before, after, delta

    legacy_u_before = data.get("u_before", {})
    legacy_u_after = data.get("u_after", {})
    legacy_delta_u = data.get("delta_u", {})
    if isinstance(legacy_u_before, dict) or isinstance(legacy_u_after, dict):
        return (
            ["bus_voltage"],
            {"bus_voltage": legacy_u_before if isinstance(legacy_u_before, dict) else {}},
            {"bus_voltage": legacy_u_after if isinstance(legacy_u_after, dict) else {}},
            {"bus_voltage": legacy_delta_u if isinstance(legacy_delta_u, dict) else {}},
        )

    return [], {}, {}, {}
def _execute_change_load_with_services(services: Dict[str, Any], instruction: dict) -> Dict[str, Any]:
    app = services["app"]
    studycase = services["studycase"]
    interpreter = services["interpreter"]
    executor = services["executor"]
    project_name = services["project_name"]

    instruction = _ensure_instruction_result_requests(instruction, user_input="")
    requested_metrics = []
    if isinstance(instruction, dict):
        requested_metrics = instruction.get("result_requests", [])
    requested_metrics = _normalize_result_requests(requested_metrics, user_input="")

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

    ldf_list = _to_py_list(studycase.GetContents("*.ComLdf", 1))
    if not ldf_list:
        ldf = studycase.CreateObject("ComLdf", "LoadFlow")
    else:
        ldf = ldf_list[0]

    try:
        ldf_result_before = ldf.Execute()
    except Exception as e:
        return {
            "status": "error",
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "error": "loadflow_before_failed",
            "details": str(e),
        }

    before_snapshot = _collect_requested_metric_snapshots(app, requested_metrics)
    values_before = before_snapshot["values"]

    try:
        _ = resolved_load.GetAttribute("plini")
    except Exception:
        return {
            "status": "error",
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "error": f"Last {getattr(resolved_load, 'loc_name', '<unknown>')} hat kein Attribut 'plini'",
        }

    try:
        execution_result = executor.execute(instruction, resolved_load)
    except Exception as e:
        return {
            "status": "error",
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "resolved_load": getattr(resolved_load, "loc_name", None),
            "error": "load_execution_failed",
            "details": str(e),
        }

    try:
        ldf_result_after = ldf.Execute()
    except Exception as e:
        return {
            "status": "error",
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "resolved_load": getattr(resolved_load, "loc_name", None),
            "execution": execution_result,
            "error": "loadflow_after_failed",
            "details": str(e),
        }

    after_snapshot = _collect_requested_metric_snapshots(app, requested_metrics)
    values_after = after_snapshot["values"]
    delta_by_metric = _build_metric_delta_payload(
        before=values_before,
        after=values_after,
        requested_metrics=requested_metrics,
    )

    data_payload: Dict[str, Any] = {
        "requested_metrics": requested_metrics,
        "metric_metadata": _build_metric_metadata(requested_metrics),
        "before": values_before,
        "after": values_after,
        "delta": delta_by_metric,
    }

    if "bus_voltage" in requested_metrics:
        data_payload["u_before"] = values_before.get("bus_voltage", {})
        data_payload["u_after"] = values_after.get("bus_voltage", {})
        data_payload["delta_u"] = delta_by_metric.get("bus_voltage", {})

    return {
        "status": "ok",
        "tool": "execute_change_load",
        "project": project_name,
        "studycase": getattr(studycase, "loc_name", None),
        "instruction": instruction,
        "resolved_load": getattr(resolved_load, "loc_name", None),
        "execution": execution_result,
        "loadflow_debug": {
            "before_execute_result": ldf_result_before,
            "after_execute_result": ldf_result_after,
            "metric_snapshot_debug": {
                "before": before_snapshot["debug"],
                "after": after_snapshot["debug"],
            },
            "before_snapshot_debug": before_snapshot["debug"].get("bus_voltage", {}),
            "after_snapshot_debug": after_snapshot["debug"].get("bus_voltage", {}),
        },
        "data": data_payload,
    }


def _summarize_powerfactory_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    result_agent = services["result_agent"]
    llm_result_agent = services["llm_result_agent"]
    project_name = services["project_name"]

    requested_metrics, before, after, delta = _extract_metric_payload_from_result_payload(result_payload)

    messages: List[str] = []
    for metric in requested_metrics:
        metric_messages = _build_metric_messages(
            metric=metric,
            before_map=before.get(metric, {}),
            after_map=after.get(metric, {}),
            delta_map=delta.get(metric, {}),
            result_agent=result_agent,
        )
        messages.extend(metric_messages)

    if not messages:
        messages = ["Es konnten keine auswertbaren Lastfluss-Ergebnisse für die angeforderten Metriken erzeugt werden."]

    summary = llm_result_agent.summarize(messages, user_input)

    return {
        "status": "ok",
        "tool": "summarize_powerfactory_result",
        "project": project_name,
        "messages": messages,
        "answer": summary,
        "requested_metrics": requested_metrics,
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
    return _summarize_powerfactory_result_with_services(services, result_payload, user_input)


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

    inventory_result = _build_topology_inventory_with_services(services, graph_result)
    if inventory_result["status"] != "ok":
        return inventory_result

    return _interpret_entity_instruction_with_services(
        services=services,
        user_input=user_input,
        inventory=inventory_result.get("inventory", {}),
    )


def interpret_switch_instruction(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    switch_inventory_result = _build_switch_inventory_from_services(services)
    if switch_inventory_result["status"] != "ok":
        return switch_inventory_result

    inventory = _build_switch_inventory_payload(switch_inventory_result.get("switches", []))

    return _interpret_switch_instruction_with_services(
        services=services,
        user_input=user_input,
        inventory=inventory,
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

    inventory_result = _build_topology_inventory_with_services(services, graph_result)
    if inventory_result["status"] != "ok":
        return inventory_result

    return _resolve_entity_from_inventory_with_services(
        services=services,
        instruction=instruction,
        inventory=inventory_result.get("inventory", {}),
        topology_graph=graph_result.get("topology_graph"),
        max_matches=max_matches,
    )


def resolve_switch_from_inventory_llm(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    switch_inventory_result = _build_switch_inventory_from_services(services)
    if switch_inventory_result["status"] != "ok":
        return switch_inventory_result

    inventory = _build_switch_inventory_payload(switch_inventory_result.get("switches", []))

    return _resolve_switch_from_inventory_llm_with_services(
        services=services,
        instruction=instruction,
        inventory=inventory,
    )


def execute_switch_operation(
    instruction: dict,
    resolution: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
    run_loadflow_after: bool = True,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    return _execute_switch_operation_with_services(
        services=services,
        instruction=instruction,
        resolution=resolution,
        run_loadflow_after=run_loadflow_after,
    )


def summarize_switch_result(
    result_payload: dict,
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    return _summarize_switch_result_with_services(
        services=services,
        result_payload=result_payload,
        user_input=user_input,
    )