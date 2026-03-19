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


# ------------------------------------------------------------------
# LLM OUTPUT MODELS FOR DATA QUERY
# ------------------------------------------------------------------
class DataQueryTypeDecision(BaseModel):
    selected_entity_type: Optional[str] = Field(default=None)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True only if the selected entity type is sufficiently safe.")


class InventoryObjectMatchDecision(BaseModel):
    selected_object_name: Optional[str] = Field(default=None)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    alternatives: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True only if the selected object is a safe unambiguous choice.")


class AttributeSelectionDecision(BaseModel):
    selected_attribute_handles: List[str] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the match decision")
    missing_context: List[str] = Field(default_factory=list)
    should_execute: bool = Field(description="True only if the selected attributes are a safe grounded match.")


# ------------------------------------------------------------------
# LIGHTWEIGHT DATA INVENTORY (NO TOPOLOGY GRAPH)
# ------------------------------------------------------------------
def _safe_get_name(obj: Any) -> Optional[str]:
    try:
        return getattr(obj, 'loc_name', None)
    except Exception:
        return None


def _safe_get_full_name(obj: Any) -> Optional[str]:
    try:
        return obj.GetFullName()
    except Exception:
        return _safe_get_name(obj)


def _safe_get_class_name(obj: Any) -> Optional[str]:
    try:
        return obj.GetClassName()
    except Exception:
        return None


def _normalize_object_entry(obj: Any, inventory_type: str) -> Dict[str, Any]:
    name = _safe_get_name(obj)
    full_name = _safe_get_full_name(obj)
    pf_class = _safe_get_class_name(obj)
    return {
        'node_id': full_name or name,
        'name': name,
        'full_name': full_name,
        'pf_class': pf_class,
        'kind': 'pf_object',
        'degree': 0,
        'inventory_type': inventory_type,
    }


def _dedupe_inventory_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        key = item.get('full_name') or item.get('name')
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda item: (str(item.get('name') or ''), str(item.get('full_name') or '')))
    return unique


def _collect_pf_objects(app: Any, patterns: List[str]) -> List[Any]:
    objects: List[Any] = []
    for pattern in patterns:
        try:
            found = app.GetCalcRelevantObjects(pattern) or []
            objects.extend(list(found))
        except Exception:
            continue
    return objects


def _build_data_inventory_from_services(services: Dict[str, Any]) -> Dict[str, Any]:
    app = services['app']
    project_name = services['project_name']

    raw_items_by_type: Dict[str, List[Dict[str, Any]]] = {
        'bus': [_normalize_object_entry(obj, 'bus') for obj in _collect_pf_objects(app, ['*.ElmTerm'])],
        'load': [_normalize_object_entry(obj, 'load') for obj in _collect_pf_objects(app, ['*.ElmLod*'])],
        'line': [_normalize_object_entry(obj, 'line') for obj in _collect_pf_objects(app, ['*.ElmLne', '*.ElmCable'])],
        'transformer': [_normalize_object_entry(obj, 'transformer') for obj in _collect_pf_objects(app, ['*.ElmTr*'])],
        'generator': [_normalize_object_entry(obj, 'generator') for obj in _collect_pf_objects(app, ['*.ElmSym', '*.ElmAsm', '*.ElmGenstat', '*.ElmPvsys', '*.ElmSgen'])],
    }

    switch_inventory_result = _build_switch_inventory_from_services(services)
    raw_items_by_type['switch'] = switch_inventory_result.get('switches', []) if switch_inventory_result.get('status') == 'ok' else []

    items_by_type: Dict[str, List[Dict[str, Any]]] = {}
    counts_by_type: Dict[str, int] = {}
    samples_by_type: Dict[str, List[Dict[str, Any]]] = {}

    for inventory_type, items in raw_items_by_type.items():
        unique = _dedupe_inventory_items(items)
        if not unique:
            continue
        items_by_type[inventory_type] = unique
        counts_by_type[inventory_type] = len(unique)
        samples_by_type[inventory_type] = [
            {
                'name': item.get('name'),
                'pf_class': item.get('pf_class'),
                'full_name': item.get('full_name'),
            }
            for item in unique[:10]
        ]

    inventory = {
        'available_types': sorted(items_by_type.keys()),
        'counts_by_type': counts_by_type,
        'items_by_type': items_by_type,
        'samples_by_type': samples_by_type,
    }

    return {
        'status': 'ok',
        'tool': 'build_data_inventory',
        'project': project_name,
        'inventory': inventory,
    }


# ------------------------------------------------------------------
# FIELD LIBRARY FOR DATA QUERY
# ------------------------------------------------------------------
PF_DATA_FIELD_LIBRARY: Dict[str, Dict[str, Dict[str, Any]]] = {
    'bus': {
        'voltage_ll': {
            'aliases': ['leiter leiter spannung', 'leiter-leiter-spannung', 'spannung ll', 'line to line voltage', 'voltage ll'],
            'attr_candidates': ['m:ul', 'm:Ul', 'm:u1l', 'm:U1l'],
            'unit': 'kV',
            'requires_loadflow': True,
            'label': 'Leiter-Leiter-Spannung',
        },
        'voltage_ln': {
            'aliases': ['leiter erde spannung', 'leiter-erde-spannung', 'spannung gegen erde', 'phase to ground voltage', 'voltage ln'],
            'attr_candidates': ['m:u', 'm:U', 'm:u1', 'm:U1'],
            'unit': 'kV',
            'requires_loadflow': True,
            'label': 'Leiter-Erde-Spannung',
        },
        'p': {
            'aliases': ['wirkleistung', 'p', 'knoten p', 'bus p'],
            'attr_candidates': ['m:Psum:bus1', 'm:Psum', 'm:P', 'c:p'],
            'unit': 'MW',
            'requires_loadflow': True,
            'label': 'Wirkleistung',
        },
        'q': {
            'aliases': ['blindleistung', 'q', 'knoten q', 'bus q'],
            'attr_candidates': ['m:Qsum:bus1', 'm:Qsum', 'm:Q', 'c:q'],
            'unit': 'MVAr',
            'requires_loadflow': True,
            'label': 'Blindleistung',
        },
        'phases': {
            'aliases': ['phasen', 'phase set'],
            'attr_candidates': ['phtech', 'phases'],
            'unit': None,
            'requires_loadflow': False,
            'label': 'Phasen',
        },
    },
    'line': {
        'loading': {
            'aliases': ['auslastung', 'belastung', 'loading', 'thermische auslastung', 'loadfactor'],
            'attr_candidates': ['loadfactor', 'c:loadfactor', 'maxload', 'c:maxload', 'c:loading', 'm:loading', 'loading', 'c:loadingmax'],
            'unit': '%',
            'requires_loadflow': True,
            'label': 'Auslastung',
        },
        'length': {
            'aliases': ['länge', 'leitungslänge', 'length'],
            'attr_candidates': ['dline', 'length', 'line_length'],
            'unit': 'km',
            'requires_loadflow': False,
            'label': 'Länge',
        },
        'type': {
            'aliases': ['typ', 'leitungstyp', 'line type', 'kabeltyp'],
            'attr_candidates': ['typ_id', 'type_id'],
            'unit': None,
            'requires_loadflow': False,
            'label': 'Typ',
        },
        'rated_current': {
            'aliases': ['nennstrom', 'rated current', 'ampacity', 'zulässiger strom'],
            'attr_candidates': ['Inom', 'inom', 'sline', 'Ithnom'],
            'unit': 'A',
            'requires_loadflow': False,
            'label': 'Nennstrom',
        },
        'laying_factor': {
            'aliases': ['verlegefaktor', 'laying factor'],
            'attr_candidates': ['frlay', 'layfac'],
            'unit': None,
            'requires_loadflow': False,
            'label': 'Verlegefaktor',
        },
        'r1': {
            'aliases': ['r1', 'positivsystem widerstand', 'widerstand mitsystem'],
            'attr_candidates': ['R1', 'r1', 'Rline'],
            'unit': 'Ohm',
            'requires_loadflow': False,
            'label': 'R1',
        },
        'x1': {
            'aliases': ['x1', 'positivsystem reaktanz', 'reaktanz mitsystem'],
            'attr_candidates': ['X1', 'x1', 'Xline'],
            'unit': 'Ohm',
            'requires_loadflow': False,
            'label': 'X1',
        },
        'r0': {
            'aliases': ['r0', 'nullsystem widerstand'],
            'attr_candidates': ['R0', 'r0'],
            'unit': 'Ohm',
            'requires_loadflow': False,
            'label': 'R0',
        },
        'x0': {
            'aliases': ['x0', 'nullsystem reaktanz'],
            'attr_candidates': ['X0', 'x0'],
            'unit': 'Ohm',
            'requires_loadflow': False,
            'label': 'X0',
        },
        'earth_factor': {
            'aliases': ['erdfaktor', 'earth factor'],
            'attr_candidates': ['fearth', 'earthfac'],
            'unit': None,
            'requires_loadflow': False,
            'label': 'Erdfaktor',
        },
    },
    'switch': {
        'state': {
            'aliases': ['zustand', 'status', 'offen geschlossen', 'open closed', 'schalterzustand'],
            'special_reader': 'switch_state',
            'unit': None,
            'requires_loadflow': False,
            'label': 'Zustand',
        },
    },
    'load': {
        'p_set': {
            'aliases': ['wirkleistung soll', 'p soll', 'plini', 'setpoint p'],
            'attr_candidates': ['plini', 'pgini', 'Pset'],
            'unit': 'MW',
            'requires_loadflow': False,
            'label': 'Wirkleistung Soll',
        },
        'q_set': {
            'aliases': ['blindleistung soll', 'q soll', 'qlini', 'setpoint q'],
            'attr_candidates': ['qlini', 'qgini', 'Qset'],
            'unit': 'MVAr',
            'requires_loadflow': False,
            'label': 'Blindleistung Soll',
        },
    },
    'transformer': {
        'type': {
            'aliases': ['typ', 'trafotyp', 'type'],
            'attr_candidates': ['typ_id', 'type_id'],
            'unit': None,
            'requires_loadflow': False,
            'label': 'Typ',
        },
        'loading': {
            'aliases': ['auslastung', 'loading'],
            'attr_candidates': ['c:loading', 'm:loading', 'loading'],
            'unit': '%',
            'requires_loadflow': True,
            'label': 'Auslastung',
        },
        'rated_power': {
            'aliases': ['nennleistung', 'rated power', 'sn'],
            'attr_candidates': ['strn', 'Snom', 'snom'],
            'unit': 'MVA',
            'requires_loadflow': False,
            'label': 'Nennleistung',
        },
    },
    'generator': {
        'p': {
            'aliases': ['wirkleistung', 'p'],
            'attr_candidates': ['m:Psum', 'm:P', 'pgini'],
            'unit': 'MW',
            'requires_loadflow': True,
            'label': 'Wirkleistung',
        },
        'q': {
            'aliases': ['blindleistung', 'q'],
            'attr_candidates': ['m:Qsum', 'm:Q', 'qgini'],
            'unit': 'MVAr',
            'requires_loadflow': True,
            'label': 'Blindleistung',
        },
        'type': {
            'aliases': ['typ', 'generator typ', 'type'],
            'attr_candidates': ['sgn', 'typ_id', 'type_id'],
            'unit': None,
            'requires_loadflow': False,
            'label': 'Typ',
        },
    },
}


def _get_available_data_fields(entity_type: str) -> Dict[str, Dict[str, Any]]:
    return PF_DATA_FIELD_LIBRARY.get(entity_type, {}) if entity_type else {}


def _build_data_field_catalog(entity_type: str) -> List[Dict[str, Any]]:
    fields = _get_available_data_fields(entity_type)
    items: List[Dict[str, Any]] = []
    for field_name, meta in fields.items():
        items.append({
            'field': field_name,
            'label': meta.get('label', field_name),
            'aliases': meta.get('aliases', []),
            'unit': meta.get('unit'),
            'requires_loadflow': bool(meta.get('requires_loadflow', False)),
        })
    items.sort(key=lambda item: item['field'])
    return items


# ------------------------------------------------------------------

# ------------------------------------------------------------------
# RAW ATTRIBUTE CANDIDATE CATALOG FOR ATTRIBUTE LISTING
# ------------------------------------------------------------------
PF_RAW_ATTRIBUTE_CATALOG: Dict[str, List[str]] = {
    'bus': [
        'm:u', 'm:ul', 'm:U', 'm:Ul', 'm:Psum', 'm:Qsum', 'm:Psum:bus1', 'm:Qsum:bus1',
        'phtech', 'uknom', 'outserv', 'cpGrid', 'iUsage', 'loc_name'
    ],
    'line': [
        'c:loading', 'm:loading', 'c:loading1', 'm:loading1', 'loading', 'Loading', 'c:loadingmax',
        'm:i', 'm:I', 'm:i1', 'm:I1', 'm:Inom', 'Inom', 'inom', 'Ithnom', 'sline',
        'dline', 'length', 'line_length', 'typ_id', 'type_id', 'frlay', 'layfac',
        'R1', 'X1', 'R0', 'X0', 'r1', 'x1', 'r0', 'x0', 'fearth', 'earthfac', 'outserv'
    ],
    'switch': [
        'on_off', 'isclosed', 'closed', 'outserv', 'typ_id', 'type_id', 'loc_name'
    ],
    'load': [
        'plini', 'qlini', 'pgini', 'qgini', 'm:Psum', 'm:Qsum', 'm:P', 'm:Q', 'outserv'
    ],
    'transformer': [
        'c:loading', 'm:loading', 'loading', 'strn', 'Snom', 'snom', 'typ_id', 'type_id', 'outserv'
    ],
    'generator': [
        'm:Psum', 'm:Qsum', 'm:P', 'm:Q', 'pgini', 'qgini', 'sgn', 'typ_id', 'type_id', 'outserv'
    ],
}


def _build_data_query_type_chain():
    parser = PydanticOutputParser(pydantic_object=DataQueryTypeDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You classify a PowerFactory data query to one supported element type.\n'
            'You may only choose an entity type from the provided available types.\n'
            'Do not invent types.\n'
            'If there is not enough information, return should_execute=false.\n'
            'Use high confidence only for a clearly grounded interpretation.\n\n'
            '{format_instructions}'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
            'Available entity types:\n{available_types}'
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser


def _build_object_match_chain():
    parser = PydanticOutputParser(pydantic_object=InventoryObjectMatchDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You resolve a user request to one exact PowerFactory object from a provided candidate list.\n'
            'You may only select a name that appears exactly in the candidate list.\n'
            'Do not invent names.\n'
            'If there is no safe unambiguous match, return selected_object_name=null and should_execute=false.\n'
            'Use high confidence only for a clearly dominant match.\n\n'
            '{format_instructions}'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
            'Entity type: {entity_type}\n\n'
            'Available object candidates:\n{object_candidates}'
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser


def _build_attribute_selection_chain():
    parser = PydanticOutputParser(pydantic_object=AttributeSelectionDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You select the best matching PowerFactory data attributes from a provided option list.\n'
            'You may only choose handles that appear exactly in the provided options.\n'
            'Prefer semantic field handles (field::<name>) when they clearly match.\n'
            'Use raw attribute handles (attr::<name>) when they are a better match to the user wording or the desired concept.\n'
            'Do not invent handles.\n'
            'Return should_execute=false if the match is not grounded enough.\n\n'
            '{format_instructions}'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
            'Entity type: {entity_type}\n'
            'Selected object: {object_name}\n\n'
            'Available attribute options:\n{attribute_options}'
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser


def _fallback_select_entity_type(user_input: str, available_types: List[str]) -> Optional[str]:
    text = _safe_lower(user_input)
    mapping = {
        'bus': ['bus', 'knoten', 'terminal'],
        'load': ['last', 'load'],
        'switch': ['schalter', 'switch', 'breaker', 'coupler'],
        'transformer': ['trafo', 'transformer'],
        'line': ['leitung', 'line', 'kabel', 'cable'],
        'generator': ['generator', 'gen'],
    }
    for entity_type, tokens in mapping.items():
        if entity_type not in available_types:
            continue
        if any(token in text for token in tokens):
            return entity_type
    if len(available_types) == 1:
        return available_types[0]
    return None


def _fallback_select_attribute_handles(user_input: str, attribute_options: List[Dict[str, Any]]) -> List[str]:
    text = _safe_lower(user_input)
    selected: List[str] = []
    for item in attribute_options:
        handle = item.get('handle')
        tokens = [
            item.get('label', ''),
            item.get('field_name', ''),
            item.get('attribute_name', ''),
            *(item.get('aliases', []) or []),
            *(item.get('candidate_attrs', []) or []),
        ]
        joined = ' | '.join(_safe_lower(str(token)) for token in tokens if token)
        if not joined:
            continue
        if any(tok and tok in text for tok in _tokenize(joined)):
            if handle:
                selected.append(handle)
    if not selected:
        for item in attribute_options:
            if item.get('handle') == 'field::state':
                if any(token in text for token in ['zustand', 'offen', 'geschlossen', 'status', 'state']):
                    selected.append('field::state')
                    break
    seen = set()
    result: List[str] = []
    for handle in selected:
        if handle not in seen:
            seen.add(handle)
            result.append(handle)
    return result[:10]


def _semantic_request_likely_needs_loadflow(user_input: str) -> bool:
    text = _safe_lower(user_input)
    tokens = ['auslastung', 'loading', 'spannung', 'voltage', 'wirkleistung', 'blindleistung', 'strom', 'current', 'lastfluss', 'load flow']
    return any(token in text for token in tokens)


def _normalize_attr_option_label(attr_name: str) -> str:
    return attr_name.replace(':', ' : ')


def _probe_raw_attribute_handle(obj: Any, attr_name: str) -> Dict[str, Any]:
    return _read_pf_attribute_candidates(obj, [attr_name])


def _list_readable_raw_attributes(obj: Any, entity_type: str) -> List[Dict[str, Any]]:
    candidate_names = list(PF_RAW_ATTRIBUTE_CATALOG.get(entity_type, []))
    for meta in _get_available_data_fields(entity_type).values():
        for attr_name in meta.get('attr_candidates', []) or []:
            if attr_name not in candidate_names:
                candidate_names.append(attr_name)

    options: List[Dict[str, Any]] = []
    seen = set()
    for attr_name in candidate_names:
        read_result = _probe_raw_attribute_handle(obj, attr_name)
        if read_result.get('status') != 'ok':
            continue
        key = f'attr::{attr_name}'
        if key in seen:
            continue
        seen.add(key)
        display_value = read_result.get('display_value')
        if display_value is None:
            display_value = read_result.get('numeric_value')
        if display_value is None:
            display_value = read_result.get('raw_value')
        options.append({
            'handle': key,
            'kind': 'raw_attribute',
            'label': _normalize_attr_option_label(attr_name),
            'attribute_name': attr_name,
            'sample_value': display_value,
            'unit': None,
            'requires_loadflow': False,
        })
    options.sort(key=lambda item: str(item.get('attribute_name') or ''))
    return options


def _probe_specific_attribute(obj: Any, attr_name: str) -> Dict[str, Any]:
    getattribute_value = None
    getattribute_error = None
    try:
        getattribute_value = obj.GetAttribute(attr_name)
    except Exception as e:
        getattribute_error = str(e)

    getattr_value = None
    getattr_error = None
    try:
        getattr_value = getattr(obj, attr_name)
    except Exception as e:
        getattr_error = str(e)

    return {
        'attribute_name': attr_name,
        'getattribute_value': _serialize_pf_value(getattribute_value),
        'getattribute_numeric': _try_numeric(getattribute_value),
        'getattribute_error': getattribute_error,
        'getattr_value': _serialize_pf_value(getattr_value),
        'getattr_numeric': _try_numeric(getattr_value),
        'getattr_error': getattr_error,
    }


def _build_line_result_attribute_debug(obj: Any) -> Dict[str, Any]:
    has_results_value = None
    has_results_error = None
    try:
        has_results_value = obj.HasResults()
    except Exception as e:
        has_results_error = str(e)

    candidates = [
        'loadfactor', 'maxload', 'c:loadfactor', 'm:loadfactor',
        'c:maxload', 'm:maxload', 'loading', 'c:loading', 'm:loading',
        'Imaxlim', 'Inom', 'Irated'
    ]

    probes = [_probe_specific_attribute(obj, attr_name) for attr_name in candidates]

    return {
        'has_results_value': has_results_value,
        'has_results_error': has_results_error,
        'specific_probes': probes,
    }


def _build_semantic_field_options(entity_type: str) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    for field_name, meta in _get_available_data_fields(entity_type).items():
        options.append({
            'handle': f'field::{field_name}',
            'kind': 'semantic_field',
            'field_name': field_name,
            'label': meta.get('label', field_name),
            'aliases': meta.get('aliases', []),
            'candidate_attrs': meta.get('attr_candidates', []),
            'special_reader': meta.get('special_reader'),
            'unit': meta.get('unit'),
            'requires_loadflow': bool(meta.get('requires_loadflow', False)),
        })
    options.sort(key=lambda item: str(item.get('field_name') or ''))
    return options


def _format_attribute_options_for_prompt(attribute_options: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in attribute_options:
        handle = item.get('handle')
        label = item.get('label') or item.get('field_name') or item.get('attribute_name') or handle
        kind = item.get('kind')
        unit = item.get('unit')
        sample_value = item.get('sample_value')
        aliases = item.get('aliases', []) or []
        candidate_attrs = item.get('candidate_attrs', []) or []
        details: List[str] = [f'kind={kind}']
        if unit:
            details.append(f'unit={unit}')
        if aliases:
            details.append('aliases=' + ', '.join(str(x) for x in aliases[:6]))
        if candidate_attrs:
            details.append('pf_candidates=' + ', '.join(str(x) for x in candidate_attrs[:6]))
        if sample_value is not None:
            details.append(f'sample={sample_value}')
        lines.append(f'- {handle}: {label} ({"; ".join(details)})')
    return '\n'.join(lines)


def _interpret_data_query_instruction_with_services(
    services: Dict[str, Any],
    user_input: str,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services['project_name']
    available_types = inventory.get('available_types', []) if isinstance(inventory, dict) else []
    if not available_types:
        return {
            'status': 'error',
            'tool': 'interpret_data_query_instruction',
            'project': project_name,
            'user_input': user_input,
            'error': 'empty_data_inventory',
            'details': 'Für die Datenabfrage stehen keine PowerFactory-Elementtypen zur Verfügung.',
        }

    llm_decision_dump: Dict[str, Any] = {}
    selected_entity_type = None
    confidence = 'low'
    rationale = 'Fallback-Auswahl verwendet.'
    missing_context: List[str] = []
    should_execute = False

    try:
        chain, parser = _build_data_query_type_chain()
        decision = chain.invoke({
            'user_input': user_input,
            'available_types': '\n'.join(f'- {item}' for item in available_types),
            'format_instructions': parser.get_format_instructions(),
        })
        llm_decision_dump = decision.model_dump()
        selected_entity_type = decision.selected_entity_type if decision.selected_entity_type in available_types else None
        confidence = decision.confidence
        rationale = decision.rationale
        missing_context = decision.missing_context or []
        should_execute = bool(decision.should_execute)
    except Exception as e:
        llm_decision_dump = {'error': str(e)}

    if selected_entity_type is None:
        selected_entity_type = _fallback_select_entity_type(user_input, available_types)
        if selected_entity_type:
            confidence = 'medium'
            should_execute = True
            rationale = 'Entity-Typ wurde per regelbasiertem Fallback erkannt.'

    if not selected_entity_type:
        missing_context.append('entity_type')

    instruction = {
        'query_type': 'element_data',
        'entity_type': selected_entity_type,
        'entity_name_raw': user_input,
        'entity_name_candidates': _build_entity_name_candidates(user_input),
        'attribute_request_text': user_input,
        'available_types': available_types,
    }

    if not should_execute or not selected_entity_type:
        return {
            'status': 'error',
            'tool': 'interpret_data_query_instruction',
            'project': project_name,
            'user_input': user_input,
            'error': 'data_query_not_safe',
            'details': 'Die Datenabfrage konnte nicht sicher genug auf einen Elementtyp aufgelöst werden.',
            'instruction': instruction,
            'llm_decision': llm_decision_dump,
            'missing_context': sorted(set(missing_context)),
        }

    return {
        'status': 'ok',
        'tool': 'interpret_data_query_instruction',
        'project': project_name,
        'user_input': user_input,
        'instruction': instruction,
        'llm_decision': llm_decision_dump,
        'confidence': confidence,
        'rationale': rationale,
    }


def _resolve_pf_object_from_inventory_llm_with_services(
    services: Dict[str, Any],
    instruction: dict,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services['project_name']
    entity_type = instruction.get('entity_type') if isinstance(instruction, dict) else None
    if not entity_type:
        return {
            'status': 'error',
            'tool': 'resolve_pf_object_from_inventory_llm',
            'project': project_name,
            'instruction': instruction,
            'error': 'missing_entity_type',
            'details': 'In der Instruction fehlt der Elementtyp.',
        }

    items_by_type = inventory.get('items_by_type', {}) if isinstance(inventory, dict) else {}
    object_candidates = items_by_type.get(entity_type, []) or []
    if not object_candidates:
        return {
            'status': 'error',
            'tool': 'resolve_pf_object_from_inventory_llm',
            'project': project_name,
            'instruction': instruction,
            'error': 'no_object_candidates',
            'details': f'Für den Elementtyp {entity_type} wurden keine Kandidaten gefunden.',
        }

    candidate_names = [item.get('name') for item in object_candidates if item.get('name')]
    if not candidate_names:
        return {
            'status': 'error',
            'tool': 'resolve_pf_object_from_inventory_llm',
            'project': project_name,
            'instruction': instruction,
            'error': 'empty_object_names',
            'details': 'Die Kandidatenliste enthält keine verwertbaren Objektnamen.',
        }

    try:
        chain, parser = _build_object_match_chain()
        decision = chain.invoke({
            'user_input': instruction.get('entity_name_raw') or '',
            'entity_type': entity_type,
            'object_candidates': '\n'.join(f'- {name}' for name in candidate_names),
            'format_instructions': parser.get_format_instructions(),
        })
        decision_dump = decision.model_dump()
    except Exception as e:
        return {
            'status': 'error',
            'tool': 'resolve_pf_object_from_inventory_llm',
            'project': project_name,
            'instruction': instruction,
            'error': 'llm_object_match_failed',
            'details': str(e),
        }

    selected_name = decision.selected_object_name
    if selected_name not in candidate_names:
        return {
            'status': 'error',
            'tool': 'resolve_pf_object_from_inventory_llm',
            'project': project_name,
            'instruction': instruction,
            'error': 'invalid_object_selection',
            'details': 'Das LLM hat keinen gültigen exakten Objektnamen aus der Kandidatenliste zurückgegeben.',
            'llm_decision': decision_dump,
            'candidate_names': candidate_names,
        }

    if not decision.should_execute or decision.confidence.lower() != 'high':
        return {
            'status': 'error',
            'tool': 'resolve_pf_object_from_inventory_llm',
            'project': project_name,
            'instruction': instruction,
            'error': 'object_match_not_safe',
            'details': 'Das LLM hat keinen ausreichend sicheren Objekttreffer gefunden.',
            'llm_decision': decision_dump,
            'candidate_names': candidate_names,
        }

    selected_match = next(item for item in object_candidates if item.get('name') == selected_name)
    return {
        'status': 'ok',
        'tool': 'resolve_pf_object_from_inventory_llm',
        'project': project_name,
        'instruction': instruction,
        'asset_query': selected_name,
        'selected_match': selected_match,
        'matches': [selected_match],
        'llm_decision': decision_dump,
        'entity_type': entity_type,
    }


# ------------------------------------------------------------------
# DATA QUERY ATTRIBUTE LISTING / EXECUTION
# ------------------------------------------------------------------
def _try_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _serialize_pf_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        if hasattr(value, 'loc_name'):
            return getattr(value, 'loc_name', None)
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return None


def _read_pf_attribute_candidates(obj: Any, attr_candidates: List[str]) -> Dict[str, Any]:
    tried: List[Dict[str, Any]] = []
    for attr_name in attr_candidates:
        raw_value = None
        source = None
        try:
            raw_value = obj.GetAttribute(attr_name)
            source = 'GetAttribute'
        except Exception:
            raw_value = None
        if raw_value is None:
            try:
                raw_value = getattr(obj, attr_name)
                source = 'getattr'
            except Exception:
                raw_value = None
        numeric_value = _try_numeric(raw_value)
        tried.append({
            'attribute': attr_name,
            'source': source,
            'raw_value': _serialize_pf_value(raw_value),
            'numeric_value': numeric_value,
        })
        if raw_value is not None:
            return {
                'status': 'ok',
                'attribute': attr_name,
                'source': source,
                'raw_value': _serialize_pf_value(raw_value),
                'numeric_value': numeric_value,
                'tried': tried,
            }
    return {
        'status': 'error',
        'error': 'attribute_not_found',
        'tried': tried,
    }


def _read_special_field(obj: Any, special_reader: str) -> Dict[str, Any]:
    if special_reader == 'switch_state':
        state_info = _read_switch_state(obj)
        if state_info.get('status') != 'ok':
            return state_info
        normalized_state = _normalize_switch_state(
            raw_value=state_info.get('raw_value'),
            source=state_info.get('state_source', ''),
        )
        return {
            'status': 'ok',
            'attribute': state_info.get('state_source'),
            'source': 'special_reader',
            'raw_value': state_info.get('raw_value'),
            'numeric_value': None,
            'display_value': normalized_state,
            'tried': [state_info],
        }
    return {
        'status': 'error',
        'error': 'unknown_special_reader',
        'details': special_reader,
    }


def _ensure_loadflow_for_data_query(studycase: Any) -> Dict[str, Any]:
    try:
        ldf_list = _to_py_list(studycase.GetContents('*.ComLdf', 1))
        if not ldf_list:
            ldf = studycase.CreateObject('ComLdf', 'LoadFlow')
        else:
            ldf = ldf_list[0]
        result = ldf.Execute()
        return {
            'executed': True,
            'loadflow_command': getattr(ldf, 'loc_name', 'LoadFlow'),
            'loadflow_result': result,
        }
    except Exception as e:
        return {
            'executed': False,
            'error': str(e),
        }


def _list_available_object_attributes_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
) -> Dict[str, Any]:
    app = services['app']
    studycase = services['studycase']
    project_name = services['project_name']

    selected_match = resolution.get('selected_match') if isinstance(resolution, dict) else None
    if not isinstance(selected_match, dict):
        return {
            'status': 'error',
            'tool': 'list_available_object_attributes',
            'project': project_name,
            'instruction': instruction,
            'error': 'missing_selected_object',
            'details': 'Es wurde kein aufgelöstes PowerFactory-Objekt übergeben.',
        }

    entity_type = instruction.get('entity_type') if isinstance(instruction, dict) else None
    full_name = selected_match.get('full_name')
    pf_object = _get_object_by_full_name(app, full_name)
    if pf_object is None:
        return {
            'status': 'error',
            'tool': 'list_available_object_attributes',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'error': 'pf_object_not_found',
            'details': f'Das aufgelöste Objekt konnte in PowerFactory nicht geladen werden: {full_name}',
        }

    loadflow_info = {'executed': False, 'reason': 'not_required_for_listing'}
    if _semantic_request_likely_needs_loadflow(instruction.get('attribute_request_text') or ''):
        loadflow_info = _ensure_loadflow_for_data_query(studycase)
        if not loadflow_info.get('executed'):
            return {
                'status': 'error',
                'tool': 'list_available_object_attributes',
                'project': project_name,
                'instruction': instruction,
                'resolution': resolution,
                'error': 'attribute_listing_loadflow_failed',
                'details': loadflow_info.get('error', 'unknown_loadflow_error'),
                'loadflow': loadflow_info,
            }

    semantic_options = _build_semantic_field_options(entity_type)
    raw_options = _list_readable_raw_attributes(pf_object, entity_type)
    attribute_options = semantic_options + raw_options

    return {
        'status': 'ok',
        'tool': 'list_available_object_attributes',
        'project': project_name,
        'studycase': getattr(studycase, 'loc_name', None),
        'instruction': instruction,
        'resolution': resolution,
        'object': {
            'name': getattr(pf_object, 'loc_name', None),
            'full_name': full_name,
            'pf_class': pf_object.GetClassName() if hasattr(pf_object, 'GetClassName') else None,
        },
        'attribute_options': attribute_options,
        'loadflow': loadflow_info,
    }


def _select_pf_object_attributes_llm_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
    attribute_listing: dict,
) -> Dict[str, Any]:
    project_name = services['project_name']
    attribute_options = attribute_listing.get('attribute_options', []) if isinstance(attribute_listing, dict) else []
    if not attribute_options:
        return {
            'status': 'error',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'error': 'empty_attribute_options',
            'details': 'Für das ausgewählte Objekt stehen keine Attributoptionen zur Verfügung.',
        }

    available_handles = [item.get('handle') for item in attribute_options if item.get('handle')]
    object_name = (attribute_listing.get('object', {}) or {}).get('name') if isinstance(attribute_listing, dict) else None
    entity_type = instruction.get('entity_type') if isinstance(instruction, dict) else None

    llm_decision_dump: Dict[str, Any] = {}
    selected_handles: List[str] = []
    confidence = 'low'
    rationale = 'Fallback-Auswahl verwendet.'
    missing_context: List[str] = []
    should_execute = False

    try:
        chain, parser = _build_attribute_selection_chain()
        decision = chain.invoke({
            'user_input': instruction.get('attribute_request_text') or instruction.get('entity_name_raw') or '',
            'entity_type': entity_type or '',
            'object_name': object_name or '',
            'attribute_options': _format_attribute_options_for_prompt(attribute_options),
            'format_instructions': parser.get_format_instructions(),
        })
        llm_decision_dump = decision.model_dump()
        selected_handles = [handle for handle in decision.selected_attribute_handles if handle in available_handles]
        confidence = decision.confidence
        rationale = decision.rationale
        missing_context = decision.missing_context or []
        should_execute = bool(decision.should_execute)
    except Exception as e:
        llm_decision_dump = {'error': str(e)}

    if not selected_handles:
        selected_handles = _fallback_select_attribute_handles(
            instruction.get('attribute_request_text') or instruction.get('entity_name_raw') or '',
            attribute_options,
        )
        if selected_handles:
            should_execute = True
            confidence = 'medium'
            rationale = 'Attributauswahl wurde per regelbasiertem Fallback getroffen.'

    if not selected_handles:
        missing_context.append('attribute_selection')

    if not should_execute or not selected_handles:
        return {
            'status': 'error',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'attribute_listing': attribute_listing,
            'error': 'attribute_selection_not_safe',
            'details': 'Die Attributauswahl konnte nicht sicher genug aufgelöst werden.',
            'llm_decision': llm_decision_dump,
            'missing_context': sorted(set(missing_context)),
        }

    instruction_out = dict(instruction)
    instruction_out['selected_attribute_handles'] = selected_handles
    return {
        'status': 'ok',
        'tool': 'select_pf_object_attributes_llm',
        'project': project_name,
        'instruction': instruction_out,
        'selected_attribute_handles': selected_handles,
        'llm_decision': llm_decision_dump,
        'confidence': confidence,
        'rationale': rationale,
    }


def _read_attribute_handle(obj: Any, entity_type: str, handle: str) -> Dict[str, Any]:
    if handle.startswith('field::'):
        field_name = handle.split('::', 1)[1]
        meta = _get_available_data_fields(entity_type).get(field_name, {})
        if not meta:
            return {
                'status': 'error',
                'error': 'unknown_field_handle',
                'handle': handle,
            }
        if 'special_reader' in meta:
            read_result = _read_special_field(obj, meta['special_reader'])
        else:
            read_result = _read_pf_attribute_candidates(obj, meta.get('attr_candidates', []))
        if read_result.get('status') == 'ok':
            display_value = read_result.get('display_value')
            if display_value is None:
                display_value = read_result.get('numeric_value')
            if display_value is None:
                display_value = read_result.get('raw_value')

            if (
                entity_type == 'line'
                and field_name == 'loading'
            ):
                numeric_value = _try_numeric(display_value)
                selected_attr = str(read_result.get('attribute') or '')
                if numeric_value is not None and selected_attr in {'loadfactor', 'c:loadfactor'}:
                    if numeric_value <= 1.5:
                        display_value = round(numeric_value * 100.0, 6)
                elif numeric_value is not None and selected_attr in {'maxload', 'c:maxload'}:
                    display_value = round(numeric_value, 6)

            return {
                'status': 'ok',
                'handle': handle,
                'field_name': field_name,
                'label': meta.get('label', field_name),
                'unit': meta.get('unit'),
                'requires_loadflow': bool(meta.get('requires_loadflow', False)),
                'value': display_value,
                'read_debug': read_result,
            }
        return {
            'status': 'error',
            'handle': handle,
            'field_name': field_name,
            'label': meta.get('label', field_name),
            'unit': meta.get('unit'),
            'requires_loadflow': bool(meta.get('requires_loadflow', False)),
            'read_debug': read_result,
        }

    if handle.startswith('attr::'):
        attr_name = handle.split('::', 1)[1]
        read_result = _read_pf_attribute_candidates(obj, [attr_name])
        if read_result.get('status') == 'ok':
            display_value = read_result.get('display_value')
            if display_value is None:
                display_value = read_result.get('numeric_value')
            if display_value is None:
                display_value = read_result.get('raw_value')
            return {
                'status': 'ok',
                'handle': handle,
                'attribute_name': attr_name,
                'label': attr_name,
                'unit': None,
                'requires_loadflow': False,
                'value': display_value,
                'read_debug': read_result,
            }
        return {
            'status': 'error',
            'handle': handle,
            'attribute_name': attr_name,
            'label': attr_name,
            'unit': None,
            'requires_loadflow': False,
            'read_debug': read_result,
        }

    return {
        'status': 'error',
        'error': 'unknown_attribute_handle_kind',
        'handle': handle,
    }


def _read_pf_object_attributes_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
) -> Dict[str, Any]:
    app = services['app']
    studycase = services['studycase']
    project_name = services['project_name']

    selected_match = resolution.get('selected_match') if isinstance(resolution, dict) else None
    if not isinstance(selected_match, dict):
        return {
            'status': 'error',
            'tool': 'read_pf_object_attributes',
            'project': project_name,
            'instruction': instruction,
            'error': 'missing_selected_object',
            'details': 'Es wurde kein aufgelöstes PowerFactory-Objekt übergeben.',
        }

    entity_type = instruction.get('entity_type') if isinstance(instruction, dict) else None
    selected_handles = instruction.get('selected_attribute_handles', []) if isinstance(instruction, dict) else []
    if not entity_type or not selected_handles:
        return {
            'status': 'error',
            'tool': 'read_pf_object_attributes',
            'project': project_name,
            'instruction': instruction,
            'error': 'missing_selected_attribute_handles',
            'details': 'In der Instruction fehlen Typ oder ausgewählte Attribute.',
        }

    full_name = selected_match.get('full_name')
    pf_object = _get_object_by_full_name(app, full_name)
    if pf_object is None:
        return {
            'status': 'error',
            'tool': 'read_pf_object_attributes',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'error': 'pf_object_not_found',
            'details': f'Das aufgelöste Objekt konnte in PowerFactory nicht geladen werden: {full_name}',
        }

    requires_loadflow = False
    for handle in selected_handles:
        if handle.startswith('field::'):
            field_name = handle.split('::', 1)[1]
            meta = _get_available_data_fields(entity_type).get(field_name, {})
            if bool(meta.get('requires_loadflow', False)):
                requires_loadflow = True
                break

    loadflow_info = {'executed': False, 'reason': 'not_required'}
    if requires_loadflow:
        loadflow_info = _ensure_loadflow_for_data_query(studycase)
        if not loadflow_info.get('executed'):
            return {
                'status': 'error',
                'tool': 'read_pf_object_attributes',
                'project': project_name,
                'instruction': instruction,
                'resolution': resolution,
                'error': 'data_query_loadflow_failed',
                'details': loadflow_info.get('error', 'unknown_loadflow_error'),
                'loadflow': loadflow_info,
            }

    values: Dict[str, Any] = {}
    field_metadata: Dict[str, Any] = {}
    field_debug: Dict[str, Any] = {}

    for handle in selected_handles:
        read_result = _read_attribute_handle(pf_object, entity_type, handle)
        field_debug[handle] = read_result
        field_metadata[handle] = {
            'label': read_result.get('label', handle),
            'unit': read_result.get('unit'),
            'requires_loadflow': bool(read_result.get('requires_loadflow', False)),
            'field_name': read_result.get('field_name'),
            'attribute_name': read_result.get('attribute_name'),
            'handle': handle,
        }
        values[handle] = read_result.get('value') if read_result.get('status') == 'ok' else None

    extra_debug: Dict[str, Any] = {}
    loading_handle_missing = (
        entity_type == 'line'
        and any(handle == 'field::loading' for handle in selected_handles)
        and values.get('field::loading') is None
    )
    if loading_handle_missing:
        all_attribute_names = sorted([name for name in dir(pf_object) if not str(name).startswith('_')])
        readable_raw_attributes = _list_readable_raw_attributes(pf_object, entity_type)
        explicit_result_debug = _build_line_result_attribute_debug(pf_object)
        extra_debug['line_loading_attribute_debug'] = {
            'reason': 'selected line loading was not readable via current semantic mapping',
            'all_attribute_names': all_attribute_names,
            'readable_raw_attributes': readable_raw_attributes,
            'explicit_result_debug': explicit_result_debug,
        }

    return {
        'status': 'ok',
        'tool': 'read_pf_object_attributes',
        'project': project_name,
        'studycase': getattr(studycase, 'loc_name', None),
        'instruction': instruction,
        'resolution': resolution,
        'entity_type': entity_type,
        'object': {
            'name': getattr(pf_object, 'loc_name', None),
            'full_name': full_name,
            'pf_class': pf_object.GetClassName() if hasattr(pf_object, 'GetClassName') else None,
        },
        'data': {
            'entity_type': entity_type,
            'selected_attribute_handles': selected_handles,
            'field_metadata': field_metadata,
            'values': values,
        },
        'loadflow': loadflow_info,
        'debug': {
            'field_reads': field_debug,
            **extra_debug,
        },
    }


def _summarize_pf_object_data_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    project_name = services['project_name']

    data = result_payload.get('data', {}) if isinstance(result_payload, dict) else {}
    values = data.get('values', {}) if isinstance(data, dict) else {}
    field_metadata = data.get('field_metadata', {}) if isinstance(data, dict) else {}
    obj = result_payload.get('object', {}) if isinstance(result_payload, dict) else {}
    loadflow = result_payload.get('loadflow', {}) if isinstance(result_payload, dict) else {}

    object_name = obj.get('name') or obj.get('full_name') or '<unbekannt>'
    pf_class = obj.get('pf_class') or '<unknown>'

    parts: List[str] = []
    messages: List[str] = []
    for handle, value in values.items():
        meta = field_metadata.get(handle, {}) if isinstance(field_metadata, dict) else {}
        label = meta.get('label', handle)
        unit = meta.get('unit')
        if value is None:
            parts.append(f'{label}: nicht verfügbar')
            messages.append(f'{label} für {object_name}: nicht verfügbar.')
        elif unit:
            parts.append(f'{label}: {value} {unit}')
            messages.append(f'{label} für {object_name}: {value} {unit}.')
        else:
            parts.append(f'{label}: {value}')
            messages.append(f'{label} für {object_name}: {value}.')

    answer = f"Daten für '{object_name}' ({pf_class}): " + '; '.join(parts) if parts else f"Für '{object_name}' konnten keine Daten gelesen werden."
    if loadflow.get('executed'):
        answer += ' Für die angefragten Ergebnisgrößen wurde zuvor ein Lastfluss gerechnet.'

    debug_payload = result_payload.get('debug', {}) if isinstance(result_payload, dict) else {}
    line_loading_debug = debug_payload.get('line_loading_attribute_debug', {}) if isinstance(debug_payload, dict) else {}
    if isinstance(line_loading_debug, dict) and line_loading_debug:
        all_attr_names = line_loading_debug.get('all_attribute_names', []) or []
        readable_raw = line_loading_debug.get('readable_raw_attributes', []) or []
        explicit_result_debug = line_loading_debug.get('explicit_result_debug', {}) or {}
        answer += ' Debug Attribute der Leitung: '
        if all_attr_names:
            answer += 'Alle Attributnamen = ' + ', '.join(str(name) for name in all_attr_names)
        if readable_raw:
            if all_attr_names:
                answer += ' | '
            answer += 'Lesbare Raw-Attribute = ' + ', '.join(
                f"{item.get('attribute_name')}={item.get('sample_value')}" for item in readable_raw
            )
        if explicit_result_debug:
            if all_attr_names or readable_raw:
                answer += ' | '
            has_results_value = explicit_result_debug.get('has_results_value')
            has_results_error = explicit_result_debug.get('has_results_error')
            probes = explicit_result_debug.get('specific_probes', []) or []
            answer += f"Resultat-Debug HasResults={has_results_value}"
            if has_results_error:
                answer += f" (Fehler: {has_results_error})"
            if probes:
                formatted = []
                for item in probes:
                    formatted.append(
                        f"{item.get('attribute_name')}: GetAttribute={item.get('getattribute_value')}"
                        f" [num={item.get('getattribute_numeric')}, err={item.get('getattribute_error')}]"
                        f"; getattr={item.get('getattr_value')}"
                        f" [num={item.get('getattr_numeric')}, err={item.get('getattr_error')}]"
                    )
                answer += ' | Gezielte Attributtests = ' + ' || '.join(formatted)

    return {
        'status': 'ok',
        'tool': 'summarize_pf_object_data_result',
        'project': project_name,
        'answer': answer,
        'messages': messages,
    }


# ------------------------------------------------------------------
# PUBLIC DATA QUERY TOOL FUNCTIONS
# ------------------------------------------------------------------
def build_data_inventory(project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services
    return _build_data_inventory_from_services(services)


def interpret_data_query_instruction(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services

    inventory_result = _build_data_inventory_from_services(services)
    if inventory_result['status'] != 'ok':
        return inventory_result

    return _interpret_data_query_instruction_with_services(
        services=services,
        user_input=user_input,
        inventory=inventory_result.get('inventory', {}),
    )


def resolve_pf_object_from_inventory_llm(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services

    inventory_result = _build_data_inventory_from_services(services)
    if inventory_result['status'] != 'ok':
        return inventory_result

    return _resolve_pf_object_from_inventory_llm_with_services(
        services=services,
        instruction=instruction,
        inventory=inventory_result.get('inventory', {}),
    )


def list_available_object_attributes(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services

    inventory_result = _build_data_inventory_from_services(services)
    if inventory_result['status'] != 'ok':
        return inventory_result

    resolution = _resolve_pf_object_from_inventory_llm_with_services(
        services=services,
        instruction=instruction,
        inventory=inventory_result.get('inventory', {}),
    )
    if resolution.get('status') != 'ok':
        return resolution

    return _list_available_object_attributes_with_services(
        services=services,
        instruction=instruction,
        resolution=resolution,
    )


def select_pf_object_attributes_llm(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services

    inventory_result = _build_data_inventory_from_services(services)
    if inventory_result['status'] != 'ok':
        return inventory_result

    resolution = _resolve_pf_object_from_inventory_llm_with_services(
        services=services,
        instruction=instruction,
        inventory=inventory_result.get('inventory', {}),
    )
    if resolution.get('status') != 'ok':
        return resolution

    listing = _list_available_object_attributes_with_services(
        services=services,
        instruction=instruction,
        resolution=resolution,
    )
    if listing.get('status') != 'ok':
        return listing

    return _select_pf_object_attributes_llm_with_services(
        services=services,
        instruction=instruction,
        resolution=resolution,
        attribute_listing=listing,
    )


def read_pf_object_attributes(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services

    inventory_result = _build_data_inventory_from_services(services)
    if inventory_result['status'] != 'ok':
        return inventory_result

    resolution = _resolve_pf_object_from_inventory_llm_with_services(
        services=services,
        instruction=instruction,
        inventory=inventory_result.get('inventory', {}),
    )
    if resolution.get('status') != 'ok':
        return resolution

    listing = _list_available_object_attributes_with_services(
        services=services,
        instruction=instruction,
        resolution=resolution,
    )
    if listing.get('status') != 'ok':
        return listing

    selected = _select_pf_object_attributes_llm_with_services(
        services=services,
        instruction=instruction,
        resolution=resolution,
        attribute_listing=listing,
    )
    if selected.get('status') != 'ok':
        return selected

    return _read_pf_object_attributes_with_services(
        services=services,
        instruction=selected.get('instruction', instruction),
        resolution=resolution,
    )


def query_pf_object_data(
    instruction: dict,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    return read_pf_object_attributes(
        instruction=instruction,
        project_name=project_name,
    )


def summarize_pf_object_data_result(
    result_payload: dict,
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services['status'] != 'ok':
        return services
    return _summarize_pf_object_data_result_with_services(
        services=services,
        result_payload=result_payload,
        user_input=user_input,
    )
