from __future__ import annotations

import subprocess
import re
from pprint import pformat
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

from cimpy.powerfactory_agent.schemas import (
    DataQueryInstruction,
    RequestedAttributeNameDecision,
    AttributeDescriptionShortlistDecision,
    AttributeDescriptionMatchDecision, 
    DataQueryTypeDecision, 
    InventoryObjectMatchDecision, 
    AttributeSelectionDecision, 
    DataSourceDecision, 
    ResultPredefinedFieldDecision, 
    TopologyEntityNameCandidatesDecision, 
    TopologyEntityTypeDecision, 
    SwitchInstructionDecision, 
    ResultRequestDecision, 
    ResultRequestRoutingDecision
)

def _to_py_list(value: Any) -> List[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []

# ------------------------------------------------------------------
# SHARED INVENTORY HELPERS (NON-TOPOLOGY)
# ------------------------------------------------------------------
PF_INVENTORY_TYPE_PATTERNS: Dict[str, List[str]] = {
    "bus": ["*.ElmTerm"],
    "load": ["*.ElmLod*"],
    "line": ["*.ElmLne", "*.ElmCable"],
    "transformer": ["*.ElmTr*"],
    "generator": ["*.ElmSym", "*.ElmAsm", "*.ElmGenstat", "*.ElmPvsys", "*.ElmSgen"],
    "switch": ["*.Sta*"],
}


def _safe_get_name(obj: Any) -> Optional[str]:
    try:
        return getattr(obj, "loc_name", None)
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


def _normalize_inventory_entry(obj: Any, inventory_type: str, kind: str = "pf_object") -> Dict[str, Any]:
    name = _safe_get_name(obj)
    full_name = _safe_get_full_name(obj)
    pf_class = _safe_get_class_name(obj)
    return {
        "node_id": full_name or name,
        "name": name,
        "full_name": full_name,
        "pf_class": pf_class,
        "kind": kind,
        "degree": 0,
        "inventory_type": inventory_type,
    }


def _dedupe_inventory_entries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        key = item.get("full_name") or item.get("name")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("full_name") or "")))
    return unique


def _collect_pf_objects_for_patterns(app: Any, patterns: List[str]) -> List[Any]:
    objects: List[Any] = []
    for pattern in patterns:
        try:
            found = app.GetCalcRelevantObjects(pattern) or []
            objects.extend(list(found))
        except Exception:
            continue
    return objects


def _build_inventory_payload(items_by_type: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    normalized_items_by_type: Dict[str, List[Dict[str, Any]]] = {}
    counts_by_type: Dict[str, int] = {}
    samples_by_type: Dict[str, List[Dict[str, Any]]] = {}

    for inventory_type, items in items_by_type.items():
        unique = _dedupe_inventory_entries(items)
        if not unique:
            continue
        normalized_items_by_type[inventory_type] = unique
        counts_by_type[inventory_type] = len(unique)
        samples_by_type[inventory_type] = [
            {
                "name": item.get("name"),
                "pf_class": item.get("pf_class"),
                "full_name": item.get("full_name"),
            }
            for item in unique[:10]
        ]

    return {
        "available_types": sorted(normalized_items_by_type.keys()),
        "counts_by_type": counts_by_type,
        "items_by_type": normalized_items_by_type,
        "samples_by_type": samples_by_type,
    }


def _collect_inventory_items_for_type(app: Any, inventory_type: str) -> List[Dict[str, Any]]:
    patterns = PF_INVENTORY_TYPE_PATTERNS.get(inventory_type, [])
    if not patterns:
        return []

    objects = _collect_pf_objects_for_patterns(app, patterns)

    if inventory_type == "switch":
        return [
            _normalize_inventory_entry(obj, "switch", kind="sta")
            for obj in objects
            if _looks_like_switch_object(obj)
        ]

    return [
        _normalize_inventory_entry(obj, inventory_type)
        for obj in objects
    ]


def _build_unified_inventory_from_services(
    services: Dict[str, Any],
    allowed_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    app = services["app"]
    project_name = services["project_name"]

    candidate_types = allowed_types or ["bus", "load", "line", "transformer", "generator", "switch"]

    items_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for inventory_type in candidate_types:
        items = _collect_inventory_items_for_type(app, inventory_type)
        if items:
            items_by_type[inventory_type] = items

    inventory = _build_inventory_payload(items_by_type)

    return {
        "status": "ok",
        "tool": "build_unified_inventory",
        "project": project_name,
        "inventory": inventory,
        "allowed_types": candidate_types,
    }

# ------------------------------------------------------------------
# POWERFACTORY CONTEXT
# ------------------------------------------------------------------
# Initialisierung von PF -> Öffnen App, Laden Projekt, Laden Studycase etc. 
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

# Initialisierung PF-Kontext + alte Agenten (Resultagent etc.) -> Refactoring möglich
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
# Liefert verfügbaren Lastenkatalog des aktiven Projekts -> Refactoring möglich (aus LLM_interpreterAgent)
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

# Interpretiert Nutzeranfrage in strukturierte Load-Instruction -> Refactoring möglich 
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

# Löst Last auf, Refactoring möglich 
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
# wandelt Text in kleingeschriebenen Text um 
def _safe_lower(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

# zerlegt Text in einfache Token 
def _tokenize(value: str) -> List[str]:
    text = _safe_lower(value)
    for ch in "\\/()[]{}:;,.!?\"'":
        text = text.replace(ch, " ")
    return [token for token in text.split() if token]

# baut Kandidaten für Topologie-Pfad
def _build_entity_name_candidates(user_input: str) -> List[str]:
    text = (user_input or "").strip()
    if not text:
        print("[DEBUG _build_entity_name_candidates] empty input")
        return []

    payload: Dict[str, Any] = {}
    error_text: Optional[str] = None

    try:
        chain, parser = _build_entity_name_candidates_chain()
        decision = chain.invoke({
            "user_input": text,
            "format_instructions": parser.get_format_instructions(),
        })

        if hasattr(decision, "model_dump"):
            payload = decision.model_dump()
        elif hasattr(decision, "dict"):
            payload = decision.dict()
        elif isinstance(decision, dict):
            payload = dict(decision)
        else:
            payload = {}
    except Exception as exc:
        error_text = str(exc)
        payload = {}

    raw_candidates = payload.get("candidate_names", []) if isinstance(payload, dict) else []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    cleaned: List[str] = []
    seen = set()

    for item in raw_candidates:
        candidate = str(item).strip()
        if not candidate:
            continue
        norm = candidate.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(candidate)

    print("[DEBUG _build_entity_name_candidates] user_input:", repr(text))
    print("[DEBUG _build_entity_name_candidates] llm_payload:", payload)
    if error_text:
        print("[DEBUG _build_entity_name_candidates] llm_error:", error_text)
    print("[DEBUG _build_entity_name_candidates] raw_candidates:", raw_candidates)
    print("[DEBUG _build_entity_name_candidates] cleaned_candidates:", cleaned[:8])

    return cleaned[:8]

# keywordbasierte Typenbestimmung für Topologieanfragen 
def _infer_entity_type_from_text(user_input: str, inventory: Dict[str, Any]) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    available_types = inventory.get("available_types", []) if isinstance(inventory, dict) else []
    if not isinstance(available_types, list):
        available_types = []

    allowed_types = [str(t).strip() for t in available_types if str(t).strip()]

    if not allowed_types:
        return None

    try:
        chain, parser = _build_topology_entity_type_chain()
        decision = chain.invoke({
            "user_input": text,
            "available_types": "\n".join(f"- {t}" for t in allowed_types),
            "format_instructions": parser.get_format_instructions(),
        })

        if hasattr(decision, "model_dump"):
            payload = decision.model_dump()
        elif hasattr(decision, "dict"):
            payload = decision.dict()
        elif isinstance(decision, dict):
            payload = dict(decision)
        else:
            payload = {}
    except Exception:
        payload = {}

    entity_type = payload.get("entity_type") if isinstance(payload, dict) else None
    should_execute = bool(payload.get("should_execute", False)) if isinstance(payload, dict) else False
    confidence = str(payload.get("confidence") or "low").strip().lower() if isinstance(payload, dict) else "low"

    if entity_type not in allowed_types:
        return None

    if not should_execute:
        return None

    if confidence not in {"high", "medium"}:
        return None

    return entity_type


# ------------------------------------------------------------------
# TOPOLOGY INVENTORY / TOPOLOGY INTERPRETATION
# ------------------------------------------------------------------
# Baut das Inventory aus dem Topologiegraphen 
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

# baut die Instruction zur Topologie-Analyse
def _interpret_entity_instruction_with_services(
    services: Dict[str, Any],
    user_input: str,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]
    entity_name_candidates = _build_entity_name_candidates(user_input)

    instruction = {
        "query_type": "neighbors",
        "entity_type": _infer_entity_type_from_text(user_input, inventory),
        "entity_name_raw": user_input,
        "entity_name_candidates": entity_name_candidates,
        "available_types": inventory.get("available_types", []),
        "debug_entity_name_candidates": entity_name_candidates,
    }

    print("[DEBUG interpret_entity_instruction] user_input:", repr(user_input))
    print("[DEBUG interpret_entity_instruction] entity_type:", instruction.get("entity_type"))
    print("[DEBUG interpret_entity_instruction] entity_name_candidates:", entity_name_candidates)

    return {
        "status": "ok",
        "tool": "interpret_entity_instruction",
        "user_input": user_input,
        "project": project_name,
        "instruction": instruction,
    }

# Wertet Topologiegraphen aus 
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
                "entity_name_candidates": raw_candidates,
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



# baut Switch-Instruction
def _interpret_switch_instruction_with_services(
    services: Dict[str, Any],
    user_input: str,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]
    available_types = inventory.get("available_types", []) if isinstance(inventory, dict) else []
    if not isinstance(available_types, list):
        available_types = []

    entity_name_candidates = _build_entity_name_candidates(user_input)

    payload: Dict[str, Any] = {}
    error_text: Optional[str] = None

    try:
        chain, parser = _build_switch_instruction_chain()
        decision = chain.invoke({
            "user_input": user_input or "",
            "available_types": "\n".join(f"- {t}" for t in available_types),
            "format_instructions": parser.get_format_instructions(),
        })

        if hasattr(decision, "model_dump"):
            payload = decision.model_dump()
        elif hasattr(decision, "dict"):
            payload = decision.dict()
        elif isinstance(decision, dict):
            payload = dict(decision)
        else:
            payload = {}
    except Exception as exc:
        error_text = str(exc)
        payload = {}

    operation = payload.get("operation") if isinstance(payload, dict) else None
    should_execute = bool(payload.get("should_execute", False)) if isinstance(payload, dict) else False
    confidence = str(payload.get("confidence") or "low").strip().lower() if isinstance(payload, dict) else "low"

    if operation not in {"open", "close", "toggle"}:
        operation = None

    instruction = {
        "query_type": "switch_operation",
        "operation": operation,
        "entity_type": "switch",
        "entity_name_raw": user_input,
        "entity_name_candidates": entity_name_candidates,
        "available_types": available_types,
        "debug_switch_decision": payload,
    }

    print("[DEBUG interpret_switch_instruction] user_input:", repr(user_input))
    print("[DEBUG interpret_switch_instruction] available_types:", available_types)
    print("[DEBUG interpret_switch_instruction] switch_decision:", payload)
    if error_text:
        print("[DEBUG interpret_switch_instruction] llm_error:", error_text)
    print("[DEBUG interpret_switch_instruction] entity_name_candidates:", entity_name_candidates)

    if not operation or not should_execute or confidence not in {"high", "medium"}:
        return {
            "status": "error",
            "tool": "interpret_switch_instruction",
            "project": project_name,
            "user_input": user_input,
            "error": "missing_switch_operation",
            "details": "Es konnte keine Schalteroperation sicher erkannt werden.",
            "instruction": instruction,
        }

    return {
        "status": "ok",
        "tool": "interpret_switch_instruction",
        "user_input": user_input,
        "project": project_name,
        "instruction": instruction,
    }


def _resolve_objects_from_inventory_llm(
    *,
    project_name: str,
    user_input: str,
    entity_type: str,
    candidate_items: List[Dict[str, Any]],
    allow_all: bool = False,
    require_high_confidence: bool = True,
) -> Dict[str, Any]:
    if not candidate_items:
        return {
            "status": "error",
            "error": "no_candidates",
            "details": f"Keine Kandidaten für entity_type={entity_type} vorhanden.",
        }

    candidate_names = [item.get("name") for item in candidate_items if item.get("name")]
    if not candidate_names:
        return {
            "status": "error",
            "error": "empty_candidate_names",
            "details": f"Die Kandidaten für entity_type={entity_type} haben keine verwertbaren Namen.",
        }

    candidate_names_for_prompt = list(candidate_names)
    if allow_all and "__ALL__" not in candidate_names_for_prompt:
        candidate_names_for_prompt.append("__ALL__")

    try:
        chain, parser = _build_object_match_chain()
        decision = chain.invoke({
            "user_input": user_input or "",
            "entity_type": entity_type,
            "object_candidates": "\n".join(f"- {name}" for name in candidate_names_for_prompt),
            "format_instructions": parser.get_format_instructions(),
        })

        if hasattr(decision, "model_dump"):
            decision_dump = decision.model_dump()
        elif hasattr(decision, "dict"):
            decision_dump = decision.dict()
        elif isinstance(decision, dict):
            decision_dump = dict(decision)
        else:
            decision_dump = {}
    except Exception as e:
        return {
            "status": "error",
            "error": "llm_object_match_failed",
            "details": str(e),
        }

    selected_name = decision_dump.get("selected_object_name")
    selection_mode = str(decision_dump.get("selection_mode") or "one").strip().lower()
    should_execute = bool(decision_dump.get("should_execute", False))
    confidence = str(decision_dump.get("confidence") or "low").strip().lower()

    if allow_all and selected_name == "__ALL__":
        if not should_execute:
            return {
                "status": "error",
                "error": "object_match_not_safe",
                "details": "Das LLM hat keinen ausreichend sicheren Treffer für '__ALL__' gefunden.",
                "candidate_names": candidate_names,
                "llm_decision": decision_dump,
            }

        if require_high_confidence and confidence != "high":
            return {
                "status": "error",
                "error": "object_match_not_confident_enough",
                "details": "Der Treffer für '__ALL__' war nicht hoch genug abgesichert.",
                "candidate_names": candidate_names,
                "llm_decision": decision_dump,
            }

        selected_matches = [item for item in candidate_items if item.get("name") in candidate_names]

        return {
            "status": "ok",
            "project": project_name,
            "asset_query": "__ALL__",
            "selected_match": selected_matches[0] if selected_matches else None,
            "selected_matches": selected_matches,
            "matches": selected_matches,
            "selected_object_names": [item.get("name") for item in selected_matches if item.get("name")],
            "selection_mode": "all",
            "llm_decision": decision_dump,
        }

    if selected_name not in candidate_names:
        return {
            "status": "error",
            "error": "invalid_object_selection",
            "details": "Das LLM hat keinen gültigen exakten Namen aus der Kandidatenliste zurückgegeben.",
            "candidate_names": candidate_names,
            "llm_decision": decision_dump,
        }

    if not should_execute:
        return {
            "status": "error",
            "error": "object_match_not_safe",
            "details": "Das LLM hat keinen ausreichend sicheren Treffer gefunden.",
            "candidate_names": candidate_names,
            "llm_decision": decision_dump,
        }

    if require_high_confidence and confidence != "high":
        return {
            "status": "error",
            "error": "object_match_not_confident_enough",
            "details": "Der Treffer war nicht hoch genug abgesichert.",
            "candidate_names": candidate_names,
            "llm_decision": decision_dump,
        }

    selected_match = next(item for item in candidate_items if item.get("name") == selected_name)

    return {
        "status": "ok",
        "project": project_name,
        "asset_query": selected_name,
        "selected_match": selected_match,
        "selected_matches": [selected_match],
        "matches": [selected_match],
        "selected_object_names": [selected_name],
        "selection_mode": selection_mode if selection_mode in {"one", "all"} else "one",
        "llm_decision": decision_dump,
    }
def _resolve_objects_from_inventory_llm_with_services(
    services: Dict[str, Any],
    instruction: dict,
    inventory: Dict[str, Any],
) -> Dict[str, Any]:
    project_name = services["project_name"]
    entity_type = instruction.get("entity_type") if isinstance(instruction, dict) else None

    if not entity_type:
        return {
            "status": "error",
            "tool": "resolve_objects_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": "missing_entity_type",
            "details": "In der Instruction fehlt der Elementtyp.",
        }

    items_by_type = inventory.get("items_by_type", {}) if isinstance(inventory, dict) else {}
    candidate_items = items_by_type.get(entity_type, []) or []

    allow_all = entity_type != "switch"

    user_input = ""
    if isinstance(instruction, dict):
        user_input = str(
            instruction.get("entity_name_raw")
            or instruction.get("request_text")
            or instruction.get("attribute_request_text")
            or ""
        )

    resolver_result = _resolve_objects_from_inventory_llm(
        project_name=project_name,
        user_input=user_input,
        entity_type=entity_type,
        candidate_items=candidate_items,
        allow_all=allow_all,
        require_high_confidence=True,
    )

    if resolver_result.get("status") != "ok":
        return {
            "status": "error",
            "tool": "resolve_objects_from_inventory_llm",
            "project": project_name,
            "instruction": instruction,
            "error": resolver_result.get("error", "object_resolution_failed"),
            "details": resolver_result.get("details", "Die Objektauflösung ist fehlgeschlagen."),
            **(
                {"llm_decision": resolver_result.get("llm_decision")}
                if resolver_result.get("llm_decision") is not None
                else {}
            ),
            **(
                {"candidate_names": resolver_result.get("candidate_names")}
                if resolver_result.get("candidate_names") is not None
                else {}
            ),
        }

    return {
        "status": "ok",
        "tool": "resolve_objects_from_inventory_llm",
        "project": project_name,
        "instruction": instruction,
        "asset_query": resolver_result.get("asset_query"),
        "selected_match": resolver_result.get("selected_match"),
        "selected_matches": resolver_result.get("selected_matches", []),
        "selected_object_names": resolver_result.get("selected_object_names", []),
        "selection_mode": resolver_result.get("selection_mode", "one"),
        "matches": resolver_result.get("matches", []),
        "llm_decision": resolver_result.get("llm_decision", {}),
        "entity_type": entity_type,
    }


# ------------------------------------------------------------------
# SWITCH EXECUTION
# ------------------------------------------------------------------
# lädt den Schalter 
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

# liest aktuellen Schalterzustand aus 
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

# mappt verschiedene PF-Zustandsrepräsentationen auf open/closed 
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

# Setzen der Schalterstellung 
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

# Helper, der Schalterstellung ändert 
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

# Ausführung der Switchänderung 
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

# Fasst Ergebnis der Schalter-Änderung zusammen 
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

# leitet aus Nutzerfrage Ergebnis-Metriken ab
def _infer_result_requests_from_user_input(user_input: str) -> List[str]:
    text = (user_input or "").strip()
    if not text:
        return list(DEFAULT_RESULT_REQUESTS)

    supported_metrics_lines: List[str] = []
    for metric_name, spec in METRIC_SPECS.items():
        aliases = spec.get("aliases", []) or []
        label = spec.get("label", metric_name)
        unit = spec.get("unit", "")
        supported_metrics_lines.append(
            f"- {metric_name}: label={label}; unit={unit}; aliases={aliases}"
        )

    payload: Dict[str, Any] = {}
    error_text: Optional[str] = None

    try:
        chain, parser = _build_result_request_chain()
        decision = chain.invoke({
            "user_input": text,
            "supported_metrics_text": "\n".join(supported_metrics_lines),
            "format_instructions": parser.get_format_instructions(),
        })

        if hasattr(decision, "model_dump"):
            payload = decision.model_dump()
        elif hasattr(decision, "dict"):
            payload = decision.dict()
        elif isinstance(decision, dict):
            payload = dict(decision)
        else:
            payload = {}
    except Exception as exc:
        error_text = str(exc)
        payload = {}

    raw_metrics = payload.get("requested_metrics", []) if isinstance(payload, dict) else []
    if not isinstance(raw_metrics, list):
        raw_metrics = []

    should_execute = bool(payload.get("should_execute", False)) if isinstance(payload, dict) else False
    confidence = str(payload.get("confidence") or "low").strip().lower() if isinstance(payload, dict) else "low"

    cleaned: List[str] = []
    for item in raw_metrics:
        metric = str(item).strip()
        if metric in METRIC_SPECS and metric not in cleaned:
            cleaned.append(metric)

    print("[DEBUG _infer_result_requests_from_user_input] user_input:", repr(text))
    print("[DEBUG _infer_result_requests_from_user_input] llm_payload:", payload)
    if error_text:
        print("[DEBUG _infer_result_requests_from_user_input] llm_error:", error_text)
    print("[DEBUG _infer_result_requests_from_user_input] cleaned_metrics:", cleaned)

    if cleaned and should_execute and confidence in {"high", "medium"}:
        return cleaned

    return list(DEFAULT_RESULT_REQUESTS)

# Entscheidung, ob nach Laständerung Ergebnis aus der Standardfunktion gelesen werden kann oder ob ein Subrequest zum vollen Auslesen notwendig ist 
def _decide_result_request_mode_from_user_input(user_input: str) -> Dict[str, Any]:
    text = (user_input or "").strip()
    if not text:
        return {
            "mode": "default_voltage",
            "requested_metrics": [],
            "result_query_text": "",
            "should_execute": True,
            "confidence": "high",
            "rationale": "Empty user input, falling back to default voltage.",
        }

    supported_metrics_lines: List[str] = []
    for metric_name, spec in METRIC_SPECS.items():
        aliases = spec.get("aliases", []) or []
        label = spec.get("label", metric_name)
        unit = spec.get("unit", "")
        supported_metrics_lines.append(
            f"- {metric_name}: label={label}; unit={unit}; aliases={aliases}"
        )

    payload: Dict[str, Any] = {}
    try:
        chain, parser = _build_result_request_routing_chain()
        decision = chain.invoke({
            "user_input": text,
            "supported_metrics_text": "\n".join(supported_metrics_lines),
            "format_instructions": parser.get_format_instructions(),
        })

        if hasattr(decision, "model_dump"):
            payload = decision.model_dump()
        elif hasattr(decision, "dict"):
            payload = decision.dict()
        elif isinstance(decision, dict):
            payload = dict(decision)
        else:
            payload = {}
    except Exception:
        payload = {}

    mode = str(payload.get("mode") or "").strip()
    requested_metrics = payload.get("requested_metrics", []) if isinstance(payload.get("requested_metrics", []), list) else []
    result_query_text = str(payload.get("result_query_text") or "").strip()
    should_execute = bool(payload.get("should_execute", False))
    confidence = str(payload.get("confidence") or "low").strip().lower()
    rationale = str(payload.get("rationale") or "").strip()

    canonical_metrics = [m for m in requested_metrics if m in METRIC_SPECS]

    if mode == "standard_metrics" and canonical_metrics and should_execute and confidence in {"high", "medium"}:
        return {
            "mode": "standard_metrics",
            "requested_metrics": canonical_metrics,
            "result_query_text": "",
            "should_execute": True,
            "confidence": confidence,
            "rationale": rationale,
        }

    if mode == "delegate_result_query" and result_query_text and should_execute and confidence in {"high", "medium"}:
        return {
            "mode": "delegate_result_query",
            "requested_metrics": [],
            "result_query_text": result_query_text,
            "should_execute": True,
            "confidence": confidence,
            "rationale": rationale,
        }

    return {
        "mode": "default_voltage",
        "requested_metrics": list(DEFAULT_RESULT_REQUESTS),
        "result_query_text": "",
        "should_execute": True,
        "confidence": "medium",
        "rationale": rationale or "No standard metric or delegated result query could be grounded safely.",
    }

# mappt Aliasnamen auf interne Metriknamen; kein Ergebnis: Weiterleitung zu infer_result_requests_from_user_input
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


# vereinheitlicht Anweisung 
def _ensure_instruction_result_requests(instruction: dict, user_input: str = "") -> dict:
    instruction_out = dict(instruction or {})

    normalized = _normalize_result_requests(
        instruction_out.get("result_requests", []),
        user_input=user_input,
    )

    inferred_from_user = _infer_result_requests_from_user_input(user_input)
    routing = _decide_result_request_mode_from_user_input(user_input)

    print("[DEBUG _ensure_instruction_result_requests] user_input:", repr(user_input))
    print("[DEBUG _ensure_instruction_result_requests] incoming instruction.result_requests:", instruction_out.get("result_requests"))
    print("[DEBUG _ensure_instruction_result_requests] normalized:", normalized)
    print("[DEBUG _ensure_instruction_result_requests] inferred_from_user:", inferred_from_user)
    print("[DEBUG _ensure_instruction_result_requests] routing:", routing)

    # Starke Priorität für klar erkannte unterstützte Standardmetriken aus dem User-Text
    if inferred_from_user and inferred_from_user != list(DEFAULT_RESULT_REQUESTS):
        instruction_out["result_requests"] = inferred_from_user
        instruction_out["result_request_mode"] = "standard_metrics"
    elif routing.get("mode") == "standard_metrics":
        instruction_out["result_requests"] = (
            routing.get("requested_metrics") or normalized or list(DEFAULT_RESULT_REQUESTS)
        )
        instruction_out["result_request_mode"] = "standard_metrics"
    elif routing.get("mode") == "delegate_result_query":
        instruction_out["result_requests"] = normalized or list(DEFAULT_RESULT_REQUESTS)
        instruction_out["result_request_mode"] = "delegate_result_query"
    else:
        instruction_out["result_requests"] = normalized or list(DEFAULT_RESULT_REQUESTS)
        instruction_out["result_request_mode"] = routing.get("mode", "default_voltage")

    instruction_out["result_query_text"] = routing.get("result_query_text", "")
    instruction_out["result_request_rationale"] = routing.get("rationale", "")
    instruction_out["result_request_confidence"] = routing.get("confidence", "low")

    return instruction_out

# Attributzugriff
def _safe_get_pf_attribute(obj: Any, attr_name: str) -> Any:
    try:
        return obj.GetAttribute(attr_name)
    except Exception:
        return None

# Auslesen der Einheit des Attributs
def _get_pf_attribute_unit(obj: Any, attr_name: str) -> Optional[str]:
    if obj is None or not attr_name:
        return None

    method_names = [
        'GetAttributeUnit',
        'GetAttributeUnits',
    ]
    for method_name in method_names:
        try:
            method = getattr(obj, method_name, None)
            if callable(method):
                unit = method(attr_name)
                if unit not in (None, ''):
                    return str(unit)
        except Exception:
            pass

    try:
        desc = obj.GetAttributeDescription(attr_name)
        if desc is not None:
            unit = getattr(desc, 'unit', None)
            if unit not in (None, ''):
                return str(unit)
    except Exception:
        pass

    unit_suffix_candidates = [
        f'{attr_name}:unit',
        f'{attr_name}.unit',
    ]
    for unit_attr in unit_suffix_candidates:
        try:
            unit = obj.GetAttribute(unit_attr)
            if unit not in (None, ''):
                return str(unit)
        except Exception:
            pass
        try:
            unit = getattr(obj, unit_attr)
            if unit not in (None, ''):
                return str(unit)
        except Exception:
            pass

    return None

# zieht name, pf_class und full_name in ein einheitliches Identitätsobjekt 
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

# überführt PF-Werte in float 
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


# versucht eine Attributliste der Reihe nach, sammelt Debug-Infos über Fehlversuche 
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

# read-Funktionen: spezialisierte metrikspezifische Leser mit Kandidatenlisten für passsende PF-Attribute 

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

# sammelt Snapshot-Werte 
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

# Auslese- / Auswerteblock für Spannungen / Grenzwerte / Metriken 

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


def _read_bus_voltage_limits_with_debug(bus: Any) -> Dict[str, Any]:
    identity = _build_pf_object_identity(bus)
    identity["bus_name"] = identity.get("name")

    tried_limits: List[Dict[str, Any]] = []
    limit_map: Dict[str, Optional[float]] = {}

    for label, attr_candidates in {
        "umin": ["umin", "u_min", "vmin"],
        "umax": ["umax", "u_max", "vmax"],
    }.items():
        read_result = _read_first_available_attribute_with_debug(bus, attr_candidates, identity)
        limit_map[label] = read_result.get("value") if read_result.get("ok") else None
        tried_limits.append({
            "limit": label,
            "used_attr": read_result.get("used_attr"),
            "tried_attrs": read_result.get("tried_attrs", []),
            "value": read_result.get("value"),
            "ok": read_result.get("ok", False),
            "non_numeric_hits": read_result.get("non_numeric_hits", []),
        })

    return {
        "ok": limit_map.get("umin") is not None or limit_map.get("umax") is not None,
        "value": limit_map,
        "tried_limits": tried_limits,
        **identity,
    }



def _snapshot_bus_voltage_limits_with_debug(app: Any) -> Dict[str, Any]:
    values: Dict[str, Dict[str, Optional[float]]] = {}
    missing: List[Dict[str, Any]] = []
    total_buses = 0
    seen_ids: set[str] = set()

    for obj in app.GetCalcRelevantObjects("*.ElmTerm") or []:
        total_buses += 1
        identity = _build_pf_object_identity(obj)
        object_id = identity.get("full_name") or identity.get("name")
        if not object_id or object_id in seen_ids:
            continue
        seen_ids.add(object_id)

        read_result = _read_bus_voltage_limits_with_debug(obj)
        bus_name = read_result.get("bus_name") or identity.get("name") or identity.get("full_name")
        if not bus_name:
            continue

        value_map = read_result.get("value") or {}
        if value_map.get("umin") is not None or value_map.get("umax") is not None:
            values[bus_name] = value_map
        else:
            missing.append({
                "bus": bus_name,
                "pf_class": identity.get("pf_class"),
                "full_name": identity.get("full_name"),
                "tried_limits": read_result.get("tried_limits", []),
            })

    return {
        "values": values,
        "debug": {
            "num_buses_total": total_buses,
            "num_buses_with_limits": len(values),
            "num_buses_missing_limits": len(missing),
            "missing_buses": missing,
            "object_queries": ["*.ElmTerm"],
        },
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

    voltage_limit_snapshot = _snapshot_bus_voltage_limits_with_debug(app)

    return {
        "values": before_data,
        "debug": snapshot_debug,
        "voltage_limits": voltage_limit_snapshot.get("values", {}),
        "voltage_limits_debug": voltage_limit_snapshot.get("debug", {}),
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
    voltage_limits: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    spec = METRIC_SPECS.get(metric, {})
    metric_label = spec.get("label", metric)
    unit = spec.get("unit", "")

    if metric == "bus_voltage" and hasattr(result_agent, "interpret_voltage_change"):
        try:
            return result_agent.interpret_voltage_change(before_map, after_map, voltage_limits=voltage_limits)
        except TypeError:
            try:
                return result_agent.interpret_voltage_change(before_map, after_map)
            except Exception:
                pass
        except Exception:
            pass

    messages: List[str] = []
    messages.append(
        f"Für {metric_label} wurden {len(before_map)} Vorher-Werte, {len(after_map)} Nachher-Werte "
        f"und {len(delta_map)} Differenzen ermittelt."
    )
    messages.extend(_build_top_delta_lines(metric_label=metric_label, delta_map=delta_map, unit=unit))
    return messages


def _extract_metric_payload_from_result_payload(result_payload: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    data = result_payload.get("data", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(data, dict):
        return [], {}, {}, {}, {}

    requested_metrics = data.get("requested_metrics")
    before = data.get("before")
    after = data.get("after")
    delta = data.get("delta")
    voltage_limits = data.get("voltage_limits", {})

    if isinstance(requested_metrics, list) and isinstance(before, dict) and isinstance(after, dict) and isinstance(delta, dict):
        return requested_metrics, before, after, delta, voltage_limits if isinstance(voltage_limits, dict) else {}

    legacy_u_before = data.get("u_before", {})
    legacy_u_after = data.get("u_after", {})
    legacy_delta_u = data.get("delta_u", {})
    if isinstance(legacy_u_before, dict) or isinstance(legacy_u_after, dict):
        return (
            ["bus_voltage"],
            {"bus_voltage": legacy_u_before if isinstance(legacy_u_before, dict) else {}},
            {"bus_voltage": legacy_u_after if isinstance(legacy_u_after, dict) else {}},
            {"bus_voltage": legacy_delta_u if isinstance(legacy_delta_u, dict) else {}},
            voltage_limits if isinstance(voltage_limits, dict) else {},
        )

    return [], {}, {}, {}, {}

# Ausführen der Laständerung und Sammeln der Metriken, Überschneidung mit Last auflösen etc. prüfen; Refactoring
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
    voltage_limits = before_snapshot.get("voltage_limits", {})

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
    # ============================================================
    # DEBUG: Bus-Spannungen und Spannungsgrenzen aus PowerFactory
    # ============================================================
    # ============================================================
    # DEBUG: Bus-Spannungen und Spannungsgrenzen aus PowerFactory
    # ============================================================
    try:
        print("\n[DEBUG] Bus-Spannungen und Spannungsgrenzen:")

        bus_voltage_before = values_before.get("bus_voltage", {}) if isinstance(values_before, dict) else {}
        bus_voltage_after = values_after.get("bus_voltage", {}) if isinstance(values_after, dict) else {}
        pf_buses = app.GetCalcRelevantObjects("*.ElmTerm") or []

        bus_objects_by_name: Dict[str, Any] = {}
        for bus_obj in pf_buses:
            try:
                bus_name = getattr(bus_obj, "loc_name", None)
            except Exception:
                bus_name = None
            if bus_name and bus_name not in bus_objects_by_name:
                bus_objects_by_name[bus_name] = bus_obj

        def _fmt_u(val: Any) -> str:
            try:
                return f"{float(val):.4f} p.u."
            except Exception:
                return str(val)

        all_bus_names = set(bus_voltage_before.keys()) | set(bus_voltage_after.keys())

        for bus_name in sorted(all_bus_names):
            u_before = bus_voltage_before.get(bus_name)
            u_after = bus_voltage_after.get(bus_name)
            bus_obj = bus_objects_by_name.get(bus_name)

            umin = None
            umax = None
            umin_attr = None
            umax_attr = None

            if bus_obj is not None:
                for candidate in ["umin", "u_min", "vmin"]:
                    try:
                        raw = bus_obj.GetAttribute(candidate)
                    except Exception:
                        raw = None
                    if raw is None:
                        try:
                            raw = getattr(bus_obj, candidate)
                        except Exception:
                            raw = None
                    if raw is not None:
                        umin = raw
                        umin_attr = candidate
                        break

                for candidate in ["umax", "u_max", "vmax"]:
                    try:
                        raw = bus_obj.GetAttribute(candidate)
                    except Exception:
                        raw = None
                    if raw is None:
                        try:
                            raw = getattr(bus_obj, candidate)
                        except Exception:
                            raw = None
                    if raw is not None:
                        umax = raw
                        umax_attr = candidate
                        break

            u_before_text = _fmt_u(u_before)
            u_after_text = _fmt_u(u_after)

            delta_text = "-"
            try:
                if u_before is not None and u_after is not None:
                    delta_text = f"{float(u_after) - float(u_before):+.4f} p.u."
            except Exception:
                pass

            print(
                f"  {bus_name}: "
                f"U_before={u_before_text} | "
                f"U_after={u_after_text} | "
                f"ΔU={delta_text} | "
                f"umin={umin} ({umin_attr}) | "
                f"umax={umax} ({umax_attr})"
            )

            if u_before is None:
                print(f"    [WARN] No U_before for {bus_name}")

        print("[DEBUG END]\n")

    except Exception as e:
        print(f"[DEBUG ERROR] Fehler beim Auslesen der Bus-Spannungen/Grenzwerte: {e}")

    except Exception as e:
        print(f"[DEBUG ERROR] Spannungsgrenzen-Debug fehlgeschlagen: {e}")

    data_payload: Dict[str, Any] = {
        "requested_metrics": requested_metrics,
        "metric_metadata": _build_metric_metadata(requested_metrics),
        "before": values_before,
        "after": values_after,
        "delta": delta_by_metric,
        "voltage_limits": voltage_limits,
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
            "voltage_limits_debug": before_snapshot.get("voltage_limits_debug", {}),
            "before_snapshot_debug": before_snapshot["debug"].get("bus_voltage", {}),
            "after_snapshot_debug": after_snapshot["debug"].get("bus_voltage", {}),
        },
        "data": data_payload,
    }

# fasst Ergebnisse der Laständerung zusammen; Refactoring
def _summarize_powerfactory_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    result_agent = services["result_agent"]
    llm_result_agent = services["llm_result_agent"]
    project_name = services["project_name"]

    requested_metrics, before, after, delta, voltage_limits = _extract_metric_payload_from_result_payload(result_payload)

    messages: List[str] = []
    for metric in requested_metrics:
        metric_messages = _build_metric_messages(
            metric=metric,
            before_map=before.get(metric, {}),
            after_map=after.get(metric, {}),
            delta_map=delta.get(metric, {}),
            result_agent=result_agent,
            voltage_limits=voltage_limits if metric == "bus_voltage" else None,
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
# FIELD LIBRARY FOR DATA QUERY - semantische Feldbibliothek für Data-Query-Pfad; Refactoring: auslagern oder LLM-basierter machen? 
# ------------------------------------------------------------------
PF_DATA_FIELD_LIBRARY: Dict[str, Dict[str, Dict[str, Any]]] = {
    'bus': {
        'nominal_voltage': {
            'aliases': ['spannung', 'basisdaten spannung', 'nennspannung', 'bus spannung', 'spannung basisdaten', 'nominal voltage'],
            'attr_candidates': ['uknom'],
            'unit': 'kV',
            'requires_loadflow': False,
            'label': 'Nennspannung',
        },
        'voltage_setpoint': {
            'aliases': ['spannungssollwert', 'sollspannung', 'u soll', 'usetp', 'setpoint voltage'],
            'attr_candidates': ['usetp', 'u0', 'uset'],
            'unit': 'p.u.',
            'requires_loadflow': False,
            'label': 'Spannungssollwert',
        },
        'voltage_upper_limit': {
            'aliases': ['obere spannungsgrenze', 'spannungsobergrenze', 'u max', 'umax', 'upper voltage limit', 'maximum voltage limit'],
            'attr_candidates': ['umax', 'u_max', 'vmax'],
            'unit': 'p.u.',
            'requires_loadflow': False,
            'label': 'Obere Spannungsgrenze',
        },
        'voltage_lower_limit': {
            'aliases': ['untere spannungsgrenze', 'spannungsuntergrenze', 'u min', 'umin', 'lower voltage limit', 'minimum voltage limit'],
            'attr_candidates': ['umin', 'u_min', 'vmin'],
            'unit': 'p.u.',
            'requires_loadflow': False,
            'label': 'Untere Spannungsgrenze',
        },
        'voltage_ll': {
            'aliases': ['leiter leiter spannung', 'leiter-leiter-spannung', 'spannung ll', 'line to line voltage', 'voltage ll', 'spannung nach lastfluss', 'lastfluss spannung'],
            'attr_candidates': ['m:ul', 'm:Ul', 'm:u1l', 'm:U1l'],
            'unit': 'p.u.',
            'requires_loadflow': True,
            'label': 'Leiter-Leiter-Spannung',
        },
        'voltage_ln': {
            'aliases': ['leiter erde spannung', 'leiter-erde-spannung', 'spannung gegen erde', 'phase to ground voltage', 'voltage ln'],
            'attr_candidates': ['m:u', 'm:U', 'm:u1', 'm:U1'],
            'unit': 'p.u.',
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
            'aliases': ['auslastung', 'belastung', 'loading', 'thermische auslastung'],
            'attr_candidates': ['c:loading', 'm:loading', 'loading', 'c:loadingmax'],
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
# RAW ATTRIBUTE CANDIDATE CATALOG FOR ATTRIBUTE LISTING - liefert bekannte Attributnamen und Heuristik für Einheiten, falls PF nichts liefert
# ------------------------------------------------------------------
PF_ATTRIBUTE_UNIT_OVERRIDES: Dict[str, str] = {
    'uknom': 'kV',
    'usetp': 'p.u.',
    'u0': 'p.u.',
    'umax': 'p.u.',
    'umin': 'p.u.',
    'm:u': 'p.u.',
    'm:ul': 'p.u.',
    'm:u1': 'p.u.',
    'm:u1l': 'p.u.',
    'm:U': 'kV',
    'm:Ul': 'kV',
    'm:U1': 'kV',
    'm:U1l': 'kV',
    'c:u': 'p.u.',
    'c:ul': 'p.u.',
    'c:U': 'kV',
    'c:Ul': 'kV',
}

PF_RAW_ATTRIBUTE_CATALOG: Dict[str, List[str]] = {
    'bus': [
        'm:u', 'm:ul', 'm:U', 'm:Ul', 'm:Psum', 'm:Qsum', 'm:Psum:bus1', 'm:Qsum:bus1',
        'phtech', 'uknom', 'usetp', 'u0', 'umax', 'umin', 'outserv', 'cpGrid', 'iUsage', 'loc_name'
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
# ------------------------------------------------------------------
# LLM-Bausteine für Data-Query-Pfad - Refactoring: mögliches Auslagern? 
# ------------------------------------------------------------------

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
            'You resolve a user request to PowerFactory objects from a provided candidate list.\n'
            'You may only select a name that appears exactly in the candidate list, or the special token __ALL__.\n'
            'Do not invent names.\n'
            'Use __ALL__ only if the request clearly refers to all available objects of the provided type, for example plural requests or requests explicitly saying all.\n'
            'If you choose __ALL__, set selection_mode=all, selected_object_name=__ALL__, and optionally copy matching object names into selected_object_names.\n'
            'If you choose one object, set selection_mode=one and selected_object_name to the exact chosen candidate name.\n'
            'If there is no safe grounded match, return selected_object_name=null and should_execute=false.\n'
            'Use high confidence only for a clearly grounded decision.\n\n'
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

def _build_entity_name_candidates_chain():
    parser = PydanticOutputParser(pydantic_object=TopologyEntityNameCandidatesDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You extract likely PowerFactory asset name candidates from a user request.\n"
            "Return a short ordered list of candidate asset names or name fragments that could match an asset in PowerFactory.\n"
            "Rules:\n"
            "- Focus only on the asset name, not the operation or surrounding question text.\n"
            "- Do not invent technical details.\n"
            "- Prefer exact-looking names if present.\n"
            "- Keep candidates unique and ordered from most likely to least likely.\n"
            "- Return at most 8 candidates.\n"
            "- If no plausible asset name can be extracted safely, return an empty list.\n\n"
            "Important naming rule:\n"
            "- If the request contains a typed asset phrase such as 'Last A', 'Load A', 'Bus 5', 'Leitung 4-5', 'Line 4-5', "
            "'Schalter 1', or 'Switch 1', keep the FULL typed phrase as the most likely candidate.\n"
            "- Do not strip away words like 'Last', 'Load', 'Bus', 'Leitung', 'Line', 'Schalter', or 'Switch' if they are part of the user-facing asset reference.\n"
            "- You may additionally return a shorter fallback fragment such as 'A' or '4-5', but only AFTER the full typed phrase.\n"
            "- Example: 'Welche Nachbarn hat Last A?' -> ['Last A', 'A']\n"
            "- Example: 'Wie hoch ist die Auslastung der Leitung 4-5?' -> ['Leitung 4-5', '4-5']\n"
            "- Example: 'Öffne Schalter 1' -> ['Schalter 1', '1']\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

def _build_topology_entity_type_chain():
    parser = PydanticOutputParser(pydantic_object=TopologyEntityTypeDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You classify a PowerFactory topology request to one supported entity type.\n"
            "You may only choose from the provided available types.\n"
            "Do not invent types.\n"
            "If there is not enough information, return entity_type=null and should_execute=false.\n"
            "Use high confidence only for a clearly grounded interpretation.\n\n"
            "Grounding rules:\n"
            "- 'Last X' or 'Load X' refers to entity type 'load'.\n"
            "- 'Bus X' refers to entity type 'bus'.\n"
            "- 'Leitung X' or 'Line X' refers to entity type 'line'.\n"
            "- 'Schalter X' or 'Switch X' refers to entity type 'switch'.\n"
            "- If the user explicitly names the asset type, prefer that exact type.\n"
            "- Do not reinterpret 'Last A' as a bus just because a bus with a similar identifier may exist.\n\n"
            "Examples:\n"
            "- 'Welche Nachbarn hat Last A?' -> entity_type=load\n"
            "- 'Welche Nachbarn hat Bus 5?' -> entity_type=bus\n"
            "- 'Welche Nachbarn hat Leitung 4-5?' -> entity_type=line\n"
            "- 'Welche Nachbarn hat Schalter 1?' -> entity_type=switch\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Available entity types:\n{available_types}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

def _build_requested_attribute_extraction_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You are a strict JSON extraction component for PowerFactory attribute-name extraction.\n'
            'Extract only attribute names that the user explicitly wrote as literal technical names.\n'
            'Do NOT infer semantic equivalents.\n'
            'Do NOT map descriptive phrases to PowerFactory names.\n'
            'Examples:\n'
            '- "m:u" -> keep\n'
            '- "uknom" -> keep\n'
            '- "Nennspannung" -> do NOT convert to "uknom"\n'
            '- "Leiter-Erde Nennspannung" -> do NOT invent an attribute name\n'
            'Return ONLY valid structured output matching the required schema.'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
        ),
    ])
    return _build_structured_chain(prompt, RequestedAttributeNameDecision)

# LLM-Prompt für Schalter-Operation
def _build_switch_instruction_chain():
    parser = PydanticOutputParser(pydantic_object=SwitchInstructionDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You interpret a PowerFactory switch operation request.\n"
            "Your task is to determine the intended switch operation.\n"
            "You may only choose one of these operations: open, close, toggle.\n"
            "If the request does not clearly specify a switch operation, return operation=null and should_execute=false.\n"
            "Use high confidence only if the operation is clearly grounded in the user request.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Available inventory types:\n{available_types}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

#LLM-Prompt für Auslesen des gesuchten Ergebnisses nach Laständerung
def _build_result_request_chain():
    parser = PydanticOutputParser(pydantic_object=ResultRequestDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You identify which PowerFactory result metrics the user is explicitly asking for.\n"
            "You may only choose from the provided supported metrics.\n"
            "Use the alias information semantically, not as exact string matching only.\n"
            "Do not invent new metrics.\n"
            "Return canonical internal metric names only.\n"
            "If no supported result metric is explicitly requested, return an empty list and should_execute=false.\n\n"

            "Very important distinction:\n"
            "- Distinguish strictly between the requested ACTION and the requested RESULT.\n"
            "- A load-change instruction such as 'Erhöhe Last A um 2 MW' or 'Reduziere Last B um 2 MW' describes the action only.\n"
            "- The change amount (for example '2 MW') is NOT automatically a requested result metric.\n"
            "- Do NOT infer result metrics from the action target or from the action magnitude alone.\n"
            "- Only return a result metric if the user explicitly asks to see, compare, display, analyze, or evaluate a result quantity.\n\n"

            "Examples:\n"
            "- 'Reduziere Last B um 2 MW' -> requested_metrics=[]\n"
            "- 'Erhöhe Last A um 2 MW' -> requested_metrics=[]\n"
            "- 'Erhöhe Last A um 2 MW. Wie verändert sich die Auslastung der Leitung 4-5?' -> requested_metrics=['line_loading']\n"
            "- 'Reduziere Last A um 2 MW und zeige die Spannungen danach' -> requested_metrics=['bus_voltage']\n"
            "- 'Wie ändern sich die Blindleistungen?' -> requested_metrics=['bus_q']\n"
            "- 'Wie ändern sich die Wirkleistungen?' -> requested_metrics=['bus_p']\n\n"

            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Supported metrics and aliases:\n{supported_metrics_text}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

# LLM-Prompt zur Entscheidung, ob nach Laständerung das Ergebnis "standardmäßig" ausgelesen werden kann oder ob ein Subrequest notwendig ist für Data Query
def _build_result_request_routing_chain():
    parser = PydanticOutputParser(pydantic_object=ResultRequestRoutingDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You decide how a PowerFactory load-change request should handle requested result values.\n"
            "Choose exactly one mode:\n"
            "- standard_metrics: the requested result can be handled by the supported standard metrics\n"
            "- default_voltage: no concrete result question is present, so default to bus voltage\n"
            "- delegate_result_query: a concrete result question is present, but it should be handled by a separate result data query\n\n"

            "Rules:\n"
            "- Use standard_metrics only if the user explicitly asks for a result quantity.\n"
            "- Use default_voltage if the user only requests the load change itself and does not explicitly ask for a result quantity.\n"
            "- The action magnitude (for example '2 MW') is part of the load-change instruction, not automatically a requested result metric.\n"
            "- Do NOT route to standard_metrics just because the action mentions MW, power, or load size.\n"
            "- Use delegate_result_query only if a concrete result question is present but does not fit the standard metrics cleanly.\n"
            "- When using standard_metrics, return canonical metric names only.\n"
            "- When using delegate_result_query, return a standalone follow-up query in result_query_text.\n\n"

            "Examples:\n"
            "- 'Reduziere Last B um 2 MW' -> mode=default_voltage, requested_metrics=[]\n"
            "- 'Erhöhe Last A um 2 MW' -> mode=default_voltage, requested_metrics=[]\n"
            "- 'Reduziere Last A um 2 MW und zeige die Spannungen danach' -> mode=standard_metrics, requested_metrics=['bus_voltage']\n"
            "- 'Erhöhe Last A um 2 MW. Wie verändert sich die Auslastung der Leitung 4-5?' -> mode=standard_metrics, requested_metrics=['line_loading']\n"
            "- 'Wie ändern sich die Blindleistungen danach?' -> mode=standard_metrics, requested_metrics=['bus_q']\n"
            "- 'Wie ändern sich die Wirkleistungen danach?' -> mode=standard_metrics, requested_metrics=['bus_p']\n\n"

            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Supported standard metrics and aliases:\n{supported_metrics_text}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

# extrahiert technische Attributnamen und normalisiert sie 
def _extract_requested_attribute_names_llm(user_input: str) -> Dict[str, Any]:
    try:
        chain, parser, chain_mode = _build_requested_attribute_extraction_chain()

        invoke_payload = {
            'user_input': user_input or '',
        }
        if parser is not None:
            invoke_payload['format_instructions'] = parser.get_format_instructions()

        decision = chain.invoke(invoke_payload)

        if hasattr(decision, 'model_dump'):
            decision_dict = decision.model_dump()
        elif hasattr(decision, 'dict'):
            decision_dict = decision.dict()
        elif isinstance(decision, dict):
            decision_dict = dict(decision)
        else:
            decision_dict = {}

        raw_names = decision_dict.get('requested_attribute_names', []) or []

        normalized: List[str] = []
        seen = set()
        for name in raw_names:
            try:
                cleaned = str(name).strip()
            except Exception:
                cleaned = ''
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)

        return {
            'status': 'ok',
            'requested_attribute_names': normalized[:20],
            'llm_decision': decision_dict,
            'chain_mode': chain_mode,
        }

    except Exception as e:
        return {
            'status': 'error',
            'requested_attribute_names': [],
            'llm_decision': {'error': str(e)},
            'chain_mode': 'failed',
        }

# ------------------------------------------------------------------
# Debug für Data Query Pfad
# ------------------------------------------------------------------

def _print_debug_block(title: str, payload: Dict[str, Any]) -> None:
    try:
        print(f"\n[DEBUG] {title}")
        print(pformat(payload, sort_dicts=False))
        print(f"[DEBUG END] {title}\n")
    except Exception as e:
        try:
            print(f"[DEBUG ERROR] {title}: {e}")
        except Exception:
            pass



# ------------------------------------------------------------------
# Data Source-Entscheidungsblock: Entscheidung, ob Basisdaten, Lastflussergebnis oder mehrdeutiger Fall vorliegt
# ------------------------------------------------------------------

def _build_data_source_decision_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            "You classify whether a PowerFactory attribute-value request targets base data, load-flow results, or is ambiguous.\n"
            "Make the decision from the user wording itself. Do not use any rule-based pre-classification.\n"
            "Return selected_data_source=base when the request clearly asks for static object data or nominal/setpoint/type/design parameters.\n"
            "Return selected_data_source=result when the request clearly asks for load-flow results, operational state after calculation, measured/calculated network state, or contains additions such as 'nach dem Lastfluss', 'bei Lastfluss', 'im Lastfluss', 'aus dem Lastfluss', 'load flow', or equivalent wording that points to a load-flow result.\n"
            "Return selected_data_source=ambiguous when the wording alone does not safely distinguish base data from load-flow results.\n"
            "Use high confidence only for clearly grounded cases.\n"
            "Phrases like maximal, maximum, minimal, Soll-, Nenn- normally lead to base data.\n"
            "Examples:\n"
            "- \"Nennspannung\" -> base\n"
            "- \"Sollspannung\" -> base\n"
            "- \"Auslastung bei Lastfluss\" -> result\n"
            "- \"Auslastung\" -> result\n"
            "- \"Auslastung in Kombination mit maximal, max, zulässig etc.\" -> base\n"
            "- \"Spannung nach dem Lastfluss\" -> result\n"
            "- \"Leitungsauslastung im Lastfluss\" -> result\n"
            "- \"Spannung\" -> result\n"
            "- \"Spannung in Kombination mit Nenn-, Soll-, zulässig, ...\" -> base\n"
            "Return ONLY valid structured output matching the required schema."
        ),
        (
            'user',
            "User request:\n{user_input}\n\n"
        ),
    ])
    return _build_structured_chain(prompt, DataSourceDecision)

def _classify_data_source_preference(user_input: str) -> Dict[str, Any]:
    try:
        chain, parser, chain_mode = _build_data_source_decision_chain()
        invoke_payload = {'user_input': user_input or ''}
        if parser is not None:
            invoke_payload['format_instructions'] = parser.get_format_instructions()
        decision = chain.invoke(invoke_payload)

        if hasattr(decision, 'model_dump'):
            decision_dict = decision.model_dump()
        elif hasattr(decision, 'dict'):
            decision_dict = decision.dict()
        elif isinstance(decision, dict):
            decision_dict = dict(decision)
        else:
            decision_dict = {}

        selected = str(decision_dict.get('selected_data_source') or 'ambiguous').strip().lower()
        if selected not in {'base', 'result', 'ambiguous'}:
            selected = 'ambiguous'

        return {
            'status': 'ok',
            'selected_data_source': selected,
            'confidence': decision_dict.get('confidence', 'low'),
            'rationale': decision_dict.get('rationale', ''),
            'should_execute': bool(decision_dict.get('should_execute', False)),
            'decision_mode': 'llm',
            'llm_decision': decision_dict,
            'chain_mode': chain_mode,
        }
    except Exception as e:
        return {
            'status': 'error',
            'selected_data_source': 'base',
            'confidence': 'low',
            'rationale': f'Data-source-Klassifikation fehlgeschlagen: {e}',
            'should_execute': False,
            'decision_mode': 'fallback_error',
            'llm_decision': {'error': str(e)},
            'chain_mode': 'failed',
        }
def _resolve_data_source_preference(user_input: str) -> Dict[str, Any]:
    decision = _classify_data_source_preference(user_input)
    selected = decision.get('selected_data_source', 'base')

    if selected not in {'base', 'result'}:
        decision['fallback_applied'] = True
        decision['fallback_reason'] = 'Ambiguous request defaults to base data in step 1.'
        decision['selected_data_source'] = 'base'
        decision['effective_data_source'] = 'base'
        decision['data_source_note'] = 'Anfrage war mehrdeutig; standardmäßig werden zunächst Basisdaten verwendet.'
        return decision

    decision['fallback_applied'] = False
    decision['effective_data_source'] = selected
    decision['data_source_note'] = (
        'Lastflussergebnisse wurden explizit oder sicher angefordert.'
        if selected == 'result'
        else 'Basisdaten werden verwendet.'
    )
    return decision


def _infer_data_source_preference(user_input: str) -> str:
    decision = _resolve_data_source_preference(user_input)
    return str(decision.get('effective_data_source') or 'base')


def _semantic_request_likely_needs_loadflow(user_input: str, source_preference: str = 'base') -> bool:
    if source_preference == 'result':
        return True
    decision = _classify_data_source_preference(user_input)
    if decision.get('selected_data_source') == 'result':
        return True
    return False

# ------------------------------------------------------------------
# Unterbau für Data-Query-Pfad: Extraktion technische Namen, Einheiten, Datenquelle für Attributnamen, Attribute aus Auswahloptionen aufbauen; Refactoring: teilweise heuristisch
# ------------------------------------------------------------------

def _normalize_attr_option_label(attr_name: str) -> str:
    return attr_name.replace(':', ' : ')


def _infer_data_source_from_attr_name(attr_name: str) -> str:
    text = _safe_lower(attr_name)
    if text.startswith('m:') or text.startswith('c:'):
        return 'result'
    return 'base'


def _attribute_name_likely_requires_loadflow(attr_name: Optional[str]) -> bool:
    if not attr_name:
        return False
    return _infer_data_source_from_attr_name(attr_name) == 'result'


def _extract_explicit_attribute_names(user_input: str) -> List[str]:
    text = user_input or ''
    result: List[str] = []
    seen = set()

    patterns = [
        r'`([^`]+)`',
        r'"([^"]+)"',
        r"'([^']+)'",
        r'\b(?:m|c):[A-Za-z0-9_:.]+\b',
        r'\b[A-Za-z][A-Za-z0-9_]*:[A-Za-z0-9_:.]+\b',
        r'\b[A-Za-z][A-Za-z0-9_]*\b',
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text):
            candidate = (match or '').strip()
            if not candidate:
                continue
            if pattern == r'\b[A-Za-z][A-Za-z0-9_]*\b':
                if not any(ch.isupper() for ch in candidate) and ':' not in candidate:
                    continue
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)

    return result[:20]


def _infer_unit_from_attribute_name(attr_name: Optional[str], default_unit: Optional[str] = None) -> Optional[str]:
    if not attr_name:
        return default_unit
    if attr_name in PF_ATTRIBUTE_UNIT_OVERRIDES:
        return PF_ATTRIBUTE_UNIT_OVERRIDES[attr_name]
    if re.search(r'(^|:)u[a-z0-9_]*$', attr_name):
        return 'p.u.'
    if re.search(r'(^|:)U[a-zA-Z0-9_]*$', attr_name):
        return 'kV'
    return default_unit



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
        pf_unit = read_result.get('pf_unit')
        heuristic_unit = _infer_unit_from_attribute_name(attr_name)
        unit = pf_unit or heuristic_unit
        unit_source = 'powerfactory' if pf_unit else ('heuristic' if heuristic_unit else None)
        options.append({
            'handle': key,
            'kind': 'raw_attribute',
            'label': _normalize_attr_option_label(attr_name),
            'attribute_name': attr_name,
            'sample_value': display_value,
            'unit': unit,
            'unit_source': unit_source,
            'pf_unit': pf_unit,
            'requires_loadflow': _attribute_name_likely_requires_loadflow(attr_name),
            'data_source': _infer_data_source_from_attr_name(attr_name),
        })
    options.sort(key=lambda item: str(item.get('attribute_name') or ''))
    return options

# ------------------------------------------------------------------
# Discovery / Fallback-Block für schwierige Attributfälle
# ------------------------------------------------------------------


def _fallback_match_full_attribute_list(
    obj: Any,
    user_input: str,
) -> List[Dict[str, Any]]:
    """
    Fallback: durchsucht ALLE verfügbaren Attribute des Objekts.
    Es werden nur EXAKTE String-Matches berücksichtigt.
    Kein fuzzy / kein LLM.
    """

    attribute_names = _discover_object_attribute_names(obj)
    if not attribute_names:
        return []

    text = (user_input or "").strip().lower()
    matches: List[Dict[str, Any]] = []

    for attr_name in attribute_names:
        if not attr_name:
            continue

        if attr_name.lower() != text:
            continue  # strikt nur exakter Match

        read_result = _probe_raw_attribute_handle(obj, attr_name)
        if read_result.get("status") != "ok":
            continue

        value = read_result.get("display_value")
        if value is None:
            value = read_result.get("numeric_value")
        if value is None:
            value = read_result.get("raw_value")

        pf_unit = read_result.get("pf_unit")

        matches.append({
            "handle": f"attr::{attr_name}",
            "attribute_name": attr_name,
            "value": value,
            "unit": pf_unit,
            "unit_source": "powerfactory" if pf_unit else None,
            "source": "fallback_full_catalog",
        })

    return matches


def _discover_object_attribute_names(obj: Any) -> List[str]:
    names: List[str] = []
    seen = set()

    def _add(name: Any) -> None:
        if not name:
            return
        try:
            name_str = str(name).strip()
        except Exception:
            return
        if not name_str or name_str.startswith('_') or name_str in seen:
            return
        seen.add(name_str)
        names.append(name_str)

    for method_name in ['GetAttributeList', 'GetAttributeNames', 'GetVariableList', 'GetVarList']:
        method = getattr(obj, method_name, None)
        if not callable(method):
            continue
        for args in [(), ('*',), (0,), (1,)]:
            try:
                result = method(*args)
            except Exception:
                continue
            for item in _to_py_list(result):
                _add(item)

    try:
        for name in dir(obj):
            if name.startswith('_'):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                value = None
            if callable(value):
                continue
            _add(name)
    except Exception:
        pass

    return names


def _build_dynamic_attr_candidates_for_field(obj: Any, field_name: str, meta: Dict[str, Any]) -> List[str]:
    discovered = _discover_object_attribute_names(obj)
    if not discovered:
        return []

    preferred: List[str] = []
    seen = set()

    explicit = [str(x).strip() for x in meta.get('attr_candidates', []) or [] if str(x).strip()]
    alias_tokens: List[str] = []
    for raw in [field_name, meta.get('label')] + list(meta.get('aliases', []) or []) + explicit:
        for token in _tokenize(str(raw or '')):
            if len(token) >= 2:
                alias_tokens.append(token)

    special_token_map = {
        'voltage_upper_limit': ['umax', 'upper', 'max', 'limit', 'voltage'],
        'voltage_lower_limit': ['umin', 'lower', 'min', 'limit', 'voltage'],
        'voltage_setpoint': ['usetp', 'uset', 'setpoint', 'soll', 'voltage'],
    }
    alias_tokens.extend(special_token_map.get(field_name, []))
    alias_tokens = list(dict.fromkeys(alias_tokens))

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            preferred.append(name)

    for candidate in explicit:
        for name in discovered:
            if name == candidate:
                _add(name)

    for name in discovered:
        lower_name = _safe_lower(name)
        if any(token and token == lower_name for token in explicit):
            _add(name)
            continue
        if any(token and token in lower_name for token in alias_tokens):
            _add(name)

    return preferred[:50]


def _read_field_with_dynamic_attr_fallback(obj: Any, field_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    dynamic_candidates = _build_dynamic_attr_candidates_for_field(obj, field_name, meta)
    if not dynamic_candidates:
        return {'status': 'error', 'error': 'no_dynamic_attr_candidates'}
    read_result = _read_pf_attribute_candidates(obj, dynamic_candidates)
    if read_result.get('status') == 'ok':
        read_result['dynamic_candidates'] = dynamic_candidates
    return read_result


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
            'unit_source': 'semantic' if meta.get('unit') else None,
            'requires_loadflow': bool(meta.get('requires_loadflow', False)),
            'data_source': 'result' if bool(meta.get('requires_loadflow', False)) else 'base',
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
        data_source = item.get('data_source')
        details: List[str] = [f'kind={kind}']
        if data_source:
            details.append(f'data_source={data_source}')
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

# löst die konkrete Anweisung (Objekt, Attribut etc.) auf 
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
    rationale = ''
    missing_context: List[str] = []
    should_execute = False

    try:
        chain, parser = _build_data_query_type_chain()
        decision = chain.invoke({
            'user_input': user_input,
            'available_types': '\n'.join(f'- {item}' for item in available_types),
            'format_instructions': parser.get_format_instructions(),
        })

        if hasattr(decision, 'model_dump'):
            llm_decision_dump = decision.model_dump()
        elif hasattr(decision, 'dict'):
            llm_decision_dump = decision.dict()
        elif isinstance(decision, dict):
            llm_decision_dump = dict(decision)
        else:
            llm_decision_dump = {}

        raw_selected_entity_type = None
        if hasattr(decision, 'selected_entity_type'):
            raw_selected_entity_type = decision.selected_entity_type
        elif isinstance(llm_decision_dump, dict):
            raw_selected_entity_type = llm_decision_dump.get('selected_entity_type')

        if raw_selected_entity_type in available_types:
            selected_entity_type = raw_selected_entity_type

        if hasattr(decision, 'confidence'):
            confidence = decision.confidence
        else:
            confidence = str(llm_decision_dump.get('confidence') or 'low')

        if hasattr(decision, 'rationale'):
            rationale = decision.rationale
        else:
            rationale = str(llm_decision_dump.get('rationale') or '')

        if hasattr(decision, 'missing_context'):
            missing_context = decision.missing_context or []
        else:
            raw_missing_context = llm_decision_dump.get('missing_context', [])
            missing_context = raw_missing_context if isinstance(raw_missing_context, list) else []

        if hasattr(decision, 'should_execute'):
            should_execute = bool(decision.should_execute)
        else:
            should_execute = bool(llm_decision_dump.get('should_execute', False))

    except Exception as e:
        llm_decision_dump = {'error': str(e)}

    if not selected_entity_type:
        missing_context.append('entity_type')

    requested_attribute_llm = _extract_requested_attribute_names_llm(user_input)
    requested_attribute_names = requested_attribute_llm.get('requested_attribute_names', [])
    data_source_decision = _resolve_data_source_preference(user_input)
    data_source_preference = data_source_decision.get('effective_data_source', 'base')
    data_source_note = data_source_decision.get('data_source_note') or (
        'Basisdaten werden verwendet.' if data_source_preference == 'base' else 'Lastflussergebnisse werden verwendet.'
    )

    instruction = {
        'query_type': 'element_data',
        'entity_type': selected_entity_type,
        'entity_name_raw': user_input,
        'entity_name_candidates': _build_entity_name_candidates(user_input),
        'attribute_request_text': user_input,
        'available_types': available_types,
        'requested_attribute_names': requested_attribute_names,
        'requested_attribute_name_extraction': requested_attribute_llm,
        'data_source_preference': data_source_preference,
        'data_source_note': data_source_note,
        'data_source_decision': data_source_decision,
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
            'requested_attribute_name_extraction': requested_attribute_llm,
            'missing_context': sorted(set(missing_context)),
        }

    return {
        'status': 'ok',
        'tool': 'interpret_data_query_instruction',
        'project': project_name,
        'user_input': user_input,
        'instruction': instruction,
        'llm_decision': llm_decision_dump,
        'requested_attribute_name_extraction': requested_attribute_llm,
        'confidence': confidence,
        'rationale': rationale,
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
        pf_unit = _get_pf_attribute_unit(obj, attr_name)
        heuristic_unit = _infer_unit_from_attribute_name(attr_name)
        unit = pf_unit or heuristic_unit
        unit_source = 'powerfactory' if pf_unit else ('heuristic' if heuristic_unit else None)
        tried.append({
            'attribute': attr_name,
            'source': source,
            'raw_value': _serialize_pf_value(raw_value),
            'numeric_value': numeric_value,
            'unit': unit,
            'unit_source': unit_source,
            'pf_unit': pf_unit,
            'data_source': _infer_data_source_from_attr_name(attr_name),
        })
        if raw_value is not None:
            return {
                'status': 'ok',
                'attribute': attr_name,
                'source': source,
                'raw_value': _serialize_pf_value(raw_value),
                'numeric_value': numeric_value,
                'unit': unit,
                'unit_source': unit_source,
                'pf_unit': pf_unit,
                'data_source': _infer_data_source_from_attr_name(attr_name),
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

# Auslesen der Attribute 
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

    source_preference = str((instruction or {}).get('data_source_preference') or 'base').strip().lower()
    if source_preference not in {'base', 'result'}:
        source_preference = 'base'

    loadflow_info = {'executed': False, 'reason': 'not_required_for_listing'}
    if source_preference == 'result':
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

    if source_preference == 'result':
        attribute_options = _build_result_attribute_options(
            entity_type=entity_type,
            semantic_options=semantic_options,
            raw_options=raw_options,
        )
    else:
        attribute_options = _build_base_attribute_options(pf_object)

    debug_payload = {
        'project': project_name,
        'entity_type': entity_type,
        'object_name': getattr(pf_object, 'loc_name', None),
        'request_text': (instruction or {}).get('attribute_request_text'),
        'data_source_decision': (instruction or {}).get('data_source_decision'),
        'data_source_preference_effective': source_preference,
        'attribute_search_mode': source_preference,
        'num_attribute_options': len(attribute_options),
        'attribute_option_preview': [
            {
                'handle': item.get('handle'),
                'label': item.get('label') or item.get('field_name') or item.get('attribute_name'),
                'data_source': item.get('data_source'),
                'requires_loadflow': bool(item.get('requires_loadflow', False)),
            }
            for item in attribute_options[:10]
        ],
        'loadflow': loadflow_info,
    }
    _print_debug_block('Attribute Listing', debug_payload)

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
        'attribute_search_mode': source_preference,
        'debug': debug_payload,
    }




def _build_base_attribute_options(obj: Any) -> List[Dict[str, Any]]:
    return _build_pf_description_attribute_options(obj)


def _build_result_attribute_options(
    entity_type: str,
    semantic_options: List[Dict[str, Any]],
    raw_options: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    seen = set()

    for item in semantic_options or []:
        if item.get('data_source') != 'result' and not bool(item.get('requires_loadflow', False)):
            continue
        handle = item.get('handle')
        if not handle or handle in seen:
            continue
        seen.add(handle)
        options.append(item)

    for item in raw_options or []:
        if item.get('data_source') != 'result' and not bool(item.get('requires_loadflow', False)):
            continue
        handle = item.get('handle')
        if not handle or handle in seen:
            continue
        seen.add(handle)
        options.append(item)

    options.sort(key=lambda item: (str(item.get('kind') or ''), str(item.get('label') or item.get('attribute_name') or item.get('field_name') or '')))
    return options


def _get_predefined_result_field_names(entity_type: Optional[str]) -> List[str]:
    mapping = {
        'bus': ['voltage_ln', 'voltage_ll', 'p', 'q'],
        'line': ['loading'],
        'transformer' : ['loading'],
    }
    return list(mapping.get(entity_type or '', []))


def _get_predefined_result_attribute_options(
    entity_type: Optional[str],
    attribute_options: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    allowed = set(_get_predefined_result_field_names(entity_type))
    if not allowed:
        return []
    result: List[Dict[str, Any]] = []
    for item in attribute_options or []:
        if item.get('kind') != 'semantic_field':
            continue
        field_name = item.get('field_name')
        if field_name in allowed:
            result.append(item)
    result.sort(key=lambda item: str(item.get('field_name') or ''))
    return result


def _build_predefined_result_field_selection_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You select the best matching predefined PowerFactory load-flow result fields from a provided option list.\n'
            'You may only return field names that appear EXACTLY in the provided predefined candidate list.\n'
            'Do not invent field names.\n'
            'Focus on the user request meaning.\n'
            'Spannung is not the same as Wirkleistung or Blindleistung.\n'
            'Wirkleistung is not the same as Blindleistung.\n'
            'For bus voltage requests: Leiter-Leiter and Leiter-Erde are different concepts and must not be confused.\n'
            'If the request does not safely identify a predefined result field, return an empty selection and should_execute=false.\n'
            'Return ONLY valid structured output matching the required schema.'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
            'Entity type: {entity_type}\n\n'
            'Available predefined result fields:\n{field_options}'
        ),
    ])
    return _build_structured_chain(prompt, ResultPredefinedFieldDecision)


def _format_predefined_result_field_options_for_prompt(attribute_options: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in attribute_options or []:
        field_name = item.get('field_name')
        label = item.get('label') or field_name
        aliases = ', '.join(str(x) for x in (item.get('aliases', []) or [])[:8])
        candidate_attrs = ', '.join(str(x) for x in (item.get('candidate_attrs', []) or [])[:8])
        unit = item.get('unit')
        details: List[str] = []
        if label:
            details.append(f'label={label}')
        if aliases:
            details.append(f'aliases={aliases}')
        if candidate_attrs:
            details.append(f'pf_candidates={candidate_attrs}')
        if unit:
            details.append(f'unit={unit}')
        lines.append(f'- {field_name}: ' + '; '.join(details))
    return '\n'.join(lines)


def _select_predefined_result_field_handles_llm(
    entity_type: Optional[str],
    request_text: str,
    attribute_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    predefined_options = _get_predefined_result_attribute_options(entity_type, attribute_options)
    available_field_names = [item.get('field_name') for item in predefined_options if item.get('field_name')]
    available_handles = {item.get('field_name'): item.get('handle') for item in predefined_options if item.get('field_name') and item.get('handle')}

    if not predefined_options:
        return {
            'status': 'error',
            'selected_attribute_handles': [],
            'selected_field_names': [],
            'matched_candidates': [],
            'should_execute': False,
            'confidence': 'low',
            'rationale': 'Keine vordefinierten Resultatfelder für diesen Entity-Typ verfügbar.',
            'llm_decision': {},
            'chain_mode': 'not_applicable',
        }

    try:
        chain, parser, chain_mode = _build_predefined_result_field_selection_chain()
        invoke_payload = {
            'user_input': request_text or '',
            'entity_type': entity_type or '',
            'field_options': _format_predefined_result_field_options_for_prompt(predefined_options),
        }
        if parser is not None:
            invoke_payload['format_instructions'] = parser.get_format_instructions()
        decision = chain.invoke(invoke_payload)

        if hasattr(decision, 'model_dump'):
            decision_dict = decision.model_dump()
        elif hasattr(decision, 'dict'):
            decision_dict = decision.dict()
        elif isinstance(decision, dict):
            decision_dict = dict(decision)
        else:
            decision_dict = {}

        selected_field_names: List[str] = []
        seen = set()
        for name in decision_dict.get('selected_field_names', []) or []:
            cleaned = str(name).strip()
            if cleaned and cleaned in available_field_names and cleaned not in seen:
                seen.add(cleaned)
                selected_field_names.append(cleaned)

        matched_candidates = [item for item in predefined_options if item.get('field_name') in selected_field_names]
        selected_attribute_handles = [available_handles[name] for name in selected_field_names if name in available_handles]

        return {
            'status': 'ok',
            'selected_attribute_handles': selected_attribute_handles,
            'selected_field_names': selected_field_names,
            'matched_candidates': matched_candidates,
            'should_execute': bool(decision_dict.get('should_execute')) and bool(selected_attribute_handles),
            'confidence': decision_dict.get('confidence', 'low'),
            'rationale': decision_dict.get('rationale', ''),
            'llm_decision': decision_dict,
            'chain_mode': chain_mode,
        }
    except Exception as e:
        return {
            'status': 'error',
            'selected_attribute_handles': [],
            'selected_field_names': [],
            'matched_candidates': [],
            'should_execute': False,
            'confidence': 'low',
            'rationale': f'LLM-Auswahl für vordefinierte Resultatfelder fehlgeschlagen: {e}',
            'llm_decision': {'error': str(e)},
            'chain_mode': 'failed',
        }


def _is_generic_bus_voltage_request(request_text: str) -> bool:
    text = _safe_lower(request_text)
    if not text:
        return False
    if 'spannung' not in text and 'voltage' not in text:
        return False
    disambiguating_tokens = [
        'leiter-erde', 'leiter erde', 'phase to ground', 'phase-ground', 'phase ground', 'ln',
        'leiter-leiter', 'leiter leiter', 'line to line', 'phase to phase', 'll',
        'wirkleistung', 'blindleistung', 'reactive', 'active power',
    ]
    return not any(token in text for token in disambiguating_tokens)


def _match_requested_result_handles_exact(
    requested_attribute_names: List[str],
    attribute_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_requests = []
    for name in requested_attribute_names or []:
        cleaned = _safe_lower(name)
        if cleaned:
            normalized_requests.append(cleaned)

    matched_options: List[Dict[str, Any]] = []
    seen = set()
    for item in attribute_options or []:
        labels = [
            item.get('attribute_name'),
            item.get('field_name'),
            item.get('label'),
            *(item.get('aliases', []) or []),
        ]
        normalized_labels = {_safe_lower(x) for x in labels if x}
        if not normalized_labels:
            continue
        if not any(req in normalized_labels for req in normalized_requests):
            continue
        handle = item.get('handle')
        if not handle or handle in seen:
            continue
        seen.add(handle)
        matched_options.append(item)

    return {
        'status': 'ok',
        'selected_attribute_handles': [item.get('handle') for item in matched_options if item.get('handle')],
        'matched_candidates': matched_options,
        'should_execute': bool(matched_options),
        'confidence': 'high' if matched_options else 'low',
        'rationale': 'Exakter Treffer auf Resultat-Attributname/-Label.' if matched_options else 'Kein exakter Treffer auf Resultat-Attributname/-Label.',
    }

#LLM-Match gegen alle vorhandenen Attribute im Data Query Pfad 
def _select_result_attribute_handles_from_all_options_llm(
    request_text: str,
    entity_type: Optional[str],
    object_name: Optional[str],
    attribute_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not attribute_options:
        return {
            'status': 'error',
            'selected_attribute_handles': [],
            'matched_candidates': [],
            'should_execute': False,
            'confidence': 'low',
            'rationale': 'Keine Resultat-Attributoptionen verfügbar.',
            'llm_decision': {},
            'chain_mode': 'not_applicable',
        }

    try:
        chain, parser = _build_attribute_selection_chain()

        invoke_payload = {
            'user_input': request_text or '',
            'entity_type': entity_type or '',
            'object_name': object_name or '',
            'attribute_options': _format_attribute_options_for_prompt(attribute_options),
        }
        if parser is not None:
            invoke_payload['format_instructions'] = parser.get_format_instructions()

        decision = chain.invoke(invoke_payload)

        if hasattr(decision, 'model_dump'):
            decision_dict = decision.model_dump()
        elif hasattr(decision, 'dict'):
            decision_dict = decision.dict()
        elif isinstance(decision, dict):
            decision_dict = dict(decision)
        else:
            decision_dict = {}

        selected_handles_raw = decision_dict.get('selected_attribute_handles', []) or []
        if not isinstance(selected_handles_raw, list):
            selected_handles_raw = []

        valid_by_handle = {
            item.get('handle'): item
            for item in attribute_options or []
            if item.get('handle')
        }

        selected_attribute_handles: List[str] = []
        matched_candidates: List[Dict[str, Any]] = []
        seen = set()

        for handle in selected_handles_raw:
            if handle in valid_by_handle and handle not in seen:
                seen.add(handle)
                selected_attribute_handles.append(handle)
                matched_candidates.append(valid_by_handle[handle])

        return {
            'status': 'ok',
            'selected_attribute_handles': selected_attribute_handles,
            'matched_candidates': matched_candidates,
            'should_execute': bool(decision_dict.get('should_execute', False)) and bool(selected_attribute_handles),
            'confidence': decision_dict.get('confidence', 'low'),
            'rationale': decision_dict.get('rationale', ''),
            'llm_decision': decision_dict,
        }

    except Exception as e:
        return {
            'status': 'error',
            'selected_attribute_handles': [],
            'matched_candidates': [],
            'should_execute': False,
            'confidence': 'low',
            'rationale': f'LLM-Fallback für Resultatattribute fehlgeschlagen: {e}',
            'llm_decision': {'error': str(e)},
        }

def _select_pf_object_result_attributes_llm_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
    attribute_listing: dict,
) -> Dict[str, Any]:
    project_name = services['project_name']
    attribute_options = list((attribute_listing.get('attribute_options') or [])) if isinstance(attribute_listing, dict) else []
    request_text = (instruction.get('attribute_request_text') or instruction.get('entity_name_raw') or '') if isinstance(instruction, dict) else ''
    requested_attribute_names = instruction.get('requested_attribute_names', []) if isinstance(instruction, dict) else []
    object_payload = (attribute_listing.get('object', {}) or {}) if isinstance(attribute_listing, dict) else {}
    object_name = object_payload.get('name')
    entity_type = (instruction or {}).get('entity_type')

    predefined_options = _get_predefined_result_attribute_options(entity_type, attribute_options)
    predefined_llm_result = _select_predefined_result_field_handles_llm(
        entity_type=entity_type,
        request_text=request_text,
        attribute_options=attribute_options,
    )

    exact_match_result = _match_requested_attribute_names_exact(
        requested_attribute_names=requested_attribute_names,
        attribute_options=attribute_options,
    )

    full_option_llm_result = _select_result_attribute_handles_from_all_options_llm(
        request_text=request_text,
        entity_type=entity_type,
        object_name=object_name,
        attribute_options=attribute_options,
    )

    selected_handles: List[str] = []
    selected_attributes: List[Dict[str, Any]] = []
    selection_mode = 'no_match'
    rationale = ''
    confidence = 'low'
    selection_notes: List[str] = []

    # ============================================================
    # 1) PREDEFINED RESULT FIELDS
    # ============================================================
    if predefined_llm_result.get('status') == 'ok' and predefined_llm_result.get('should_execute'):
        selected_handles = list(predefined_llm_result.get('selected_attribute_handles', []))
        selected_attributes = list(predefined_llm_result.get('matched_candidates', []))
        selection_mode = 'predefined_result_fields_llm'
        rationale = predefined_llm_result.get('rationale') or ''
        confidence = predefined_llm_result.get('confidence', 'low')

    # ============================================================
    # 2) GENERISCHE BUS-SPANNUNG IM RESULT-PFAD
    # ============================================================
    elif entity_type == 'bus' and _is_generic_bus_voltage_request(request_text):
        fallback_attr = next(
            (item for item in predefined_options if item.get('field_name') == 'voltage_ll' and item.get('handle')),
            None
        )
        if fallback_attr is not None:
            selected_handles = [fallback_attr.get('handle')]
            selected_attributes = [fallback_attr]
            selection_mode = 'result_voltage_default_ll'
            confidence = 'medium'
            rationale = 'Allgemeine Spannungsanfrage im Result-Pfad ohne Leiter-Leiter/Leiter-Erde-Angabe; standardmäßig wurde Leiter-Leiter gewählt.'
            selection_notes.append('generic_bus_voltage_default')

    # ============================================================
    # 3) EXAKTER MATCH AUF TECHNISCHEN NAMEN / LABEL
    # ============================================================
    elif exact_match_result.get('status') == 'ok' and exact_match_result.get('should_execute'):
        selected_handles = list(exact_match_result.get('selected_attribute_handles', []))
        selected_attributes = list(exact_match_result.get('matched_candidates', []))
        selection_mode = 'exact_result_attribute_name_match'
        rationale = exact_match_result.get('rationale') or ''
        confidence = exact_match_result.get('confidence', 'low')

    # ============================================================
    # 4) SEMANTISCHES LLM-MATCH GEGEN ALLE RESULT-ATTRIBUTE
    # ============================================================
    elif full_option_llm_result.get('status') == 'ok' and full_option_llm_result.get('should_execute'):
        selected_handles = list(full_option_llm_result.get('selected_attribute_handles', []))
        selected_attributes = list(full_option_llm_result.get('matched_candidates', []))
        selection_mode = 'result_attribute_options_llm'
        rationale = full_option_llm_result.get('rationale') or ''
        confidence = full_option_llm_result.get('confidence', 'low')
        selection_notes.append('semantic_result_attribute_fallback')

    instruction_out = dict(instruction or {})
    instruction_out['selected_attribute_handles'] = selected_handles

    selection_debug = {
        'project': project_name,
        'entity_type': entity_type,
        'object_name': object_name,
        'request_text': request_text,
        'requested_attribute_names': requested_attribute_names,
        'requested_attribute_name_extraction': instruction.get('requested_attribute_name_extraction'),
        'available_result_attribute_count': len(attribute_options),
        'predefined_result_attribute_count': len(predefined_options),
        'predefined_result_attribute_handles_preview': [
            item.get('handle') for item in predefined_options[:10] if item.get('handle')
        ],
        'predefined_llm_result': predefined_llm_result,
        'exact_match_result': exact_match_result,
        'full_option_llm_result': full_option_llm_result,
        'selection_mode': selection_mode,
        'selection_notes': selection_notes,
        'final_selected_handles': selected_handles,
        'final_rationale': rationale,
        'final_should_execute': bool(selected_handles),
        'final_confidence': confidence,
    }

    _print_debug_block('Result Attribute Selection', selection_debug)

    if selected_handles:
        instruction_out['attribute_selection_debug'] = selection_debug
        return {
            'status': 'ok',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction_out,
            'resolution': resolution,
            'attribute_listing': attribute_listing,
            'selected_attribute_handles': selected_handles,
            'selected_attributes': selected_attributes,
            'llm_decision': {
                'path': selection_mode,
                'predefined_llm_result': predefined_llm_result,
                'exact_match_result': exact_match_result,
                'full_option_llm_result': full_option_llm_result,
            },
            'selection_debug': selection_debug,
        }

    return {
        'status': 'error',
        'tool': 'select_pf_object_attributes_llm',
        'project': project_name,
        'instruction': instruction_out,
        'resolution': resolution,
        'attribute_listing': attribute_listing,
        'error': 'result_attribute_selection_not_safe',
        'details': 'Die Resultat-Attributauswahl konnte nicht sicher genug aufgelöst werden.',
        'selection_debug': selection_debug,
    }


def _read_pf_object_result_attributes_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
) -> Dict[str, Any]:
    instruction_out = dict(instruction or {})
    instruction_out['data_source_preference'] = 'result'
    return _read_pf_object_attributes_with_services(
        services=services,
        instruction=instruction_out,
        resolution=resolution,
    )

def _score_semantic_attribute_option(user_input: str, item: Dict[str, Any]) -> int:
    text = _safe_lower(user_input)
    if not text:
        return 0

    score = 0
    label = _safe_lower(item.get('label'))
    field_name = _safe_lower(item.get('field_name'))
    aliases = [_safe_lower(alias) for alias in (item.get('aliases', []) or [])]
    candidate_attrs = [_safe_lower(attr) for attr in (item.get('candidate_attrs', []) or [])]

    for token in [label, field_name]:
        if token and token in text:
            score += 6

    for alias in aliases:
        if alias and alias in text:
            score += 8

    for candidate_attr in candidate_attrs:
        if candidate_attr and candidate_attr in text:
            score += 5

    field_tokens = []
    for raw in [label, field_name, *aliases]:
        field_tokens.extend(_tokenize(raw))
    for token in set(field_tokens):
        if token and token in text:
            score += 1

    if item.get('kind') == 'semantic_field':
        score += 1

    return score


def _select_semantic_attribute_handles_from_request(
    user_input: str,
    attribute_options: List[Dict[str, Any]],
) -> List[str]:
    scored: List[Tuple[int, str]] = []

    for item in attribute_options:
        handle = item.get('handle')
        if not handle or not str(handle).startswith('field::'):
            continue
        score = _score_semantic_attribute_option(user_input, item)
        if score <= 0:
            continue
        scored.append((score, handle))

    if not scored:
        return []

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score = scored[0][0]
    if best_score < 6:
        return []

    selected = [handle for score, handle in scored if score == best_score]
    return selected[:10]


def _build_attribute_candidate_suggestions(
    attribute_options: List[Dict[str, Any]],
    handles: List[str],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    seen = set()
    for handle in handles:
        if not handle or handle in seen:
            continue
        seen.add(handle)
        item = next((opt for opt in attribute_options if opt.get('handle') == handle), None)
        if not item:
            continue
        suggestions.append({
            'handle': handle,
            'label': item.get('label') or item.get('field_name') or item.get('attribute_name') or handle,
            'kind': item.get('kind'),
            'attribute_name': item.get('attribute_name'),
            'field_name': item.get('field_name'),
            'unit': item.get('unit'),
            'data_source': item.get('data_source'),
            'requires_loadflow': bool(item.get('requires_loadflow', False)),
            'aliases': item.get('aliases', []) or [],
            'candidate_attrs': item.get('candidate_attrs', []) or [],
            'sample_value': item.get('sample_value'),
        })
        if len(suggestions) >= max_items:
            break
    return suggestions


def _safe_get_pf_attribute_description_text(obj: Any, attr_name: str) -> Optional[str]:
    if obj is None or not attr_name:
        return None

    try:
        desc = obj.GetAttributeDescription(attr_name)
    except Exception:
        desc = None

    if desc is None:
        return None

    if isinstance(desc, str):
        cleaned = desc.strip()
        return cleaned or None

    parts: List[str] = []
    for key in ['short', 'short_text', 'description', 'text', 'label', 'name']:
        try:
            value = getattr(desc, key, None)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            value_text = str(value).strip()
        except Exception:
            value_text = ''
        if value_text and value_text not in parts:
            parts.append(value_text)

    if parts:
        return ' | '.join(parts)

    try:
        value_text = str(desc).strip()
    except Exception:
        value_text = ''
    return value_text or None


def _build_pf_description_attribute_options(obj: Any) -> List[Dict[str, Any]]:
    attribute_names = _discover_object_attribute_names(obj)
    options: List[Dict[str, Any]] = []
    seen_handles = set()

    for attr_name in attribute_names:
        if not attr_name:
            continue
        handle = f'attr::{attr_name}'
        if handle in seen_handles:
            continue
        seen_handles.add(handle)

        read_result = _probe_raw_attribute_handle(obj, attr_name)
        description_text = _safe_get_pf_attribute_description_text(obj, attr_name)
        pf_unit = read_result.get('pf_unit') if isinstance(read_result, dict) else None
        heuristic_unit = _infer_unit_from_attribute_name(attr_name)
        unit = pf_unit or heuristic_unit
        unit_source = 'powerfactory' if pf_unit else ('heuristic' if heuristic_unit else None)

        sample_value = None
        if isinstance(read_result, dict) and read_result.get('status') == 'ok':
            sample_value = read_result.get('display_value')
            if sample_value is None:
                sample_value = read_result.get('numeric_value')
            if sample_value is None:
                sample_value = read_result.get('raw_value')

        options.append({
            'handle': handle,
            'kind': 'pf_description_attribute',
            'attribute_name': attr_name,
            'label': _normalize_attr_option_label(attr_name),
            'attribute_description': description_text,
            'sample_value': sample_value,
            'unit': unit,
            'unit_source': unit_source,
            'pf_unit': pf_unit,
            'requires_loadflow': _attribute_name_likely_requires_loadflow(attr_name),
            'data_source': _infer_data_source_from_attr_name(attr_name),
            'readable': bool(isinstance(read_result, dict) and read_result.get('status') == 'ok'),
        })

    options.sort(key=lambda item: str(item.get('attribute_name') or ''))
    return options


def _format_pf_description_options_for_prompt(attribute_options: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in attribute_options:
        attr_name = item.get('attribute_name') or item.get('handle') or '<unknown>'
        description = item.get('attribute_description') or '<keine Beschreibung>'
        lines.append(
            f'- ID: {attr_name}\n'
            f'  DESCRIPTION: {description}'
        )
    return '\n'.join(lines)


def _fallback_shortlist_pf_attributes_by_description(
    attribute_options: List[Dict[str, Any]],
    user_input: str,
    max_candidates: int = 5,
) -> Dict[str, Any]:
    text = _safe_lower(user_input)
    query_tokens = [t for t in _tokenize(text) if len(t) >= 3]

    stopwords = {
        'der', 'die', 'das', 'dem', 'den', 'des', 'ein', 'eine', 'einer', 'eines',
        'und', 'oder', 'mit', 'von', 'für', 'auf', 'im', 'in', 'am', 'an', 'zu',
        'ist', 'sind', 'was', 'welche', 'welcher', 'welches', 'line'
    }
    query_tokens = [t for t in query_tokens if t not in stopwords]

    def _score(item: Dict[str, Any]) -> tuple[float, int, str]:
        description = _safe_lower(item.get('attribute_description') or '')
        attr_name = str(item.get('attribute_name') or '')
        score = 0.0
        matched_tokens = 0

        for token in query_tokens:
            if token and token in description:
                matched_tokens += 1
                score += 2.0

        if any('therm' in token for token in query_tokens) and 'thermal' in description:
            score += 4.0
        if any(('auslast' in token) or ('belast' in token) for token in query_tokens) and 'loading' in description:
            score += 5.0
        if any('max' in token for token in query_tokens) and ('maximum' in description or 'max' in description):
            score += 2.0
        if any('spann' in token for token in query_tokens) and 'voltage' in description:
            score += 3.0
        if any('strom' in token for token in query_tokens) and 'current' in description:
            score += 3.0
        if any('nenn' in token for token in query_tokens) and ('nominal' in description or 'rated' in description):
            score += 3.0
        if any('grenz' in token for token in query_tokens) and 'limit' in description:
            score += 3.0
        if any('soll' in token for token in query_tokens) and 'setpoint' in description:
            score += 3.0
        if any('max' in token for token in query_tokens) and 'max' in attr_name.lower():
            score += 1.0

        return score, matched_tokens, attr_name

    ranked = []
    for item in attribute_options or []:
        attr_name = item.get('attribute_name')
        if not attr_name:
            continue
        score, matched_tokens, attr_name = _score(item)
        if score <= 0:
            continue
        ranked.append((score, matched_tokens, attr_name, item))

    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    shortlisted = [name for _, _, name, _ in ranked[:max_candidates]]

    return {
        'shortlisted_attribute_names': shortlisted,
        'confidence': 'low',
        'rationale': 'Heuristische Fallback-Shortlist auf Basis der attribute_description, weil die strukturierte LLM-Antwort nicht parsebar war.',
        'missing_context': [],
        'should_execute': bool(shortlisted),
    }

def _build_structured_chain(prompt: ChatPromptTemplate, schema_model: Any):
    """
    Prefer native structured output if the LLM wrapper supports it.
    Fall back to PydanticOutputParser otherwise.
    """
    llm = get_llm()

    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(schema_model)
            return prompt | structured_llm, None, "native_structured_output"
        except Exception:
            pass

    parser = PydanticOutputParser(pydantic_object=schema_model)
    return prompt | llm | parser, parser, "pydantic_output_parser"



def _build_pf_attribute_description_shortlist_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You are a strict JSON extraction component for PowerFactory attribute shortlisting.\n'
            'Your ONLY task is to shortlist candidate attribute names from the provided list.\n'
            'You are NOT allowed to answer the user question.\n'
            'You are NOT allowed to explain electrical concepts.\n'
            'You are NOT allowed to request additional context.\n'
            'Return ONLY valid structured output matching the required schema.\n'
            'Use ONLY the DESCRIPTION text for semantic matching.\n'
            'IGNORE the attribute ID/name for semantic interpretation.\n'
            'Treat the attribute ID only as the identifier you return.\n'
            'Do NOT use naming patterns, prefixes, abbreviations, units, readability flags, sample values, or data-source hints.\n'
            'Compare the FULL user request against the provided descriptions only.\n'
            'Select 3 to 8 candidate attribute names only if their descriptions are genuinely plausible matches.\n'
            'If nothing is grounded enough in the provided descriptions, return an empty shortlist and should_execute=false.\n'
            'Do not invent attribute names.'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
            'Entity type: {entity_type}\n'
            'Selected object: {object_name}\n\n'
            'Available PowerFactory attribute descriptions:\n{attribute_options}'
        ),
    ])
    return _build_structured_chain(prompt, AttributeDescriptionShortlistDecision)


def _build_pf_attribute_description_match_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            'You are a strict JSON extraction component for PowerFactory attribute matching.\n'
            'Your ONLY task is to perform the FINAL disambiguation against the provided shortlist.\n'
            'You are NOT allowed to answer the user question.\n'
            'You are NOT allowed to explain electrical concepts.\n'
            'You are NOT allowed to produce tables, markdown, prose, commentary, derivations, or examples.\n'
            'Return ONLY valid structured output matching the required schema.\n'
            'You may only select attribute names that appear EXACTLY in the provided shortlist.\n'
            'Do not invent names.\n'
            'Choose an attribute only if it is clearly better than the alternatives.\n'
            'You must consider the FULL user request, not only one keyword.\n'
            'Be especially careful with conflicts such as Leiter-Erde vs Leiter-Leiter, nominal/base vs result/load-flow values, and setpoint vs measured values.\n'
            'If multiple candidates remain plausible or the request conflicts with the shortlist, return should_execute=false.'
        ),
        (
            'user',
            'User request:\n{user_input}\n\n'
            'Entity type: {entity_type}\n'
            'Selected object: {object_name}\n\n'
            'Shortlisted real PowerFactory attributes with descriptions:\n{attribute_options}'
        ),
    ])
    return _build_structured_chain(prompt, AttributeDescriptionMatchDecision)

def _shortlist_pf_attributes_by_description_with_llm(
    obj: Any,
    entity_type: str,
    object_name: str,
    user_input: str,
    source_preference: str = 'base',
) -> Dict[str, Any]:
    attribute_options = _build_pf_description_attribute_options(obj)
    if not attribute_options:
        return {
            'status': 'error',
            'error': 'empty_pf_description_attribute_options',
            'attribute_options': [],
        }

    filtered_options = list(attribute_options)

    try:
        chain, parser, chain_mode = _build_pf_attribute_description_shortlist_chain()
        invoke_payload = {
            'user_input': user_input or '',
            'entity_type': entity_type or '',
            'object_name': object_name or '',
            'attribute_options': _format_pf_description_options_for_prompt(filtered_options),
        }
        if parser is not None:
            invoke_payload['format_instructions'] = parser.get_format_instructions()

        decision = chain.invoke(invoke_payload)
    except Exception as e:
        fallback_decision = _fallback_shortlist_pf_attributes_by_description(
            attribute_options=filtered_options,
            user_input=user_input,
            max_candidates=5,
        )
        shortlisted_names = [
            name for name in fallback_decision.get('shortlisted_attribute_names', []) or []
            if name in {item.get('attribute_name') for item in filtered_options if item.get('attribute_name')}
        ]
        shortlisted_options = [
            item for item in filtered_options
            if item.get('attribute_name') in shortlisted_names
        ]
        return {
            'status': 'ok',
            'llm_decision': fallback_decision,
            'shortlisted_attribute_names': shortlisted_names,
            'shortlisted_options': shortlisted_options,
            'attribute_options': filtered_options,
            'chain_mode': 'heuristic_fallback_after_parse_error',
            'fallback_trigger': 'pf_attribute_description_shortlist_failed',
            'fallback_details': str(e),
        }

    shortlisted_names: List[str] = []
    seen = set()
    valid_names = {item.get('attribute_name') for item in filtered_options if item.get('attribute_name')}

    for name in decision.shortlisted_attribute_names or []:
        try:
            cleaned = str(name).strip()
        except Exception:
            cleaned = ''
        if not cleaned or cleaned not in valid_names or cleaned in seen:
            continue
        seen.add(cleaned)
        shortlisted_names.append(cleaned)

    shortlisted_options = [
        item for item in filtered_options
        if item.get('attribute_name') in shortlisted_names
    ]

    return {
        'status': 'ok',
        'llm_decision': decision.model_dump(),
        'shortlisted_attribute_names': shortlisted_names,
        'shortlisted_options': shortlisted_options,
        'attribute_options': filtered_options,
        'chain_mode': chain_mode,
    }


def _match_pf_attributes_by_description_with_llm(
    obj: Any,
    entity_type: str,
    object_name: str,
    user_input: str,
    source_preference: str = 'base',
) -> Dict[str, Any]:
    shortlist_result = _shortlist_pf_attributes_by_description_with_llm(
        obj=obj,
        entity_type=entity_type,
        object_name=object_name,
        user_input=user_input,
        source_preference=source_preference,
    )

    if shortlist_result.get('status') != 'ok':
        return shortlist_result

    shortlisted_options = shortlist_result.get('shortlisted_options', []) or []
    if not shortlisted_options:
        return {
            'status': 'ok',
            'llm_decision': {
                'selected_attribute_names': [],
                'confidence': 'low',
                'rationale': 'Die Shortlist-Stufe hat keine ausreichend plausiblen PF-Attribute gefunden.',
                'missing_context': ['attribute_selection'],
                'should_execute': False,
            },
            'selected_attribute_names': [],
            'selected_attribute_handles': [],
            'matched_candidates': [],
            'attribute_options': shortlist_result.get('attribute_options', []),
            'shortlist': shortlist_result,
        }

    try:
        chain, parser, chain_mode = _build_pf_attribute_description_match_chain()
        invoke_payload = {
            'user_input': user_input or '',
            'entity_type': entity_type or '',
            'object_name': object_name or '',
            'attribute_options': _format_pf_description_options_for_prompt(shortlisted_options),
        }
        if parser is not None:
            invoke_payload['format_instructions'] = parser.get_format_instructions()

        decision = chain.invoke(invoke_payload)
    except Exception as e:
        return {
            'status': 'error',
            'error': 'pf_attribute_description_match_failed',
            'details': str(e),
            'attribute_options': shortlisted_options,
            'shortlist': shortlist_result,
        }

    selected_names: List[str] = []
    seen = set()
    valid_names = {item.get('attribute_name') for item in shortlisted_options if item.get('attribute_name')}

    for name in decision.selected_attribute_names or []:
        try:
            cleaned = str(name).strip()
        except Exception:
            cleaned = ''
        if not cleaned or cleaned not in valid_names or cleaned in seen:
            continue
        seen.add(cleaned)
        selected_names.append(cleaned)

    selected_handles = [f'attr::{name}' for name in selected_names]
    matched_candidates = [item for item in shortlisted_options if item.get('attribute_name') in selected_names]

    return {
        'status': 'ok',
        'llm_decision': decision.model_dump(),
        'selected_attribute_names': selected_names,
        'selected_attribute_handles': selected_handles,
        'matched_candidates': matched_candidates,
        'attribute_options': shortlisted_options,
        'shortlist': shortlist_result,
        'chain_mode': chain_mode,
    }

def _match_requested_attribute_names_exact(
    requested_attribute_names: List[str],
    attribute_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    selected_attribute_names: List[str] = []
    selected_attribute_handles: List[str] = []
    matched_candidates: List[Dict[str, Any]] = []
    seen_names = set()

    valid_by_name: Dict[str, Dict[str, Any]] = {}
    for item in attribute_options or []:
        attr_name = item.get('attribute_name')
        handle = item.get('handle')
        if not attr_name or not handle:
            continue
        valid_by_name[str(attr_name)] = item

    for raw_name in requested_attribute_names or []:
        if not isinstance(raw_name, str):
            continue
        if raw_name in valid_by_name and raw_name not in seen_names:
            seen_names.add(raw_name)
            item = valid_by_name[raw_name]
            selected_attribute_names.append(raw_name)
            selected_attribute_handles.append(item['handle'])
            matched_candidates.append(item)

    return {
        'status': 'ok',
        'selected_attribute_names': selected_attribute_names,
        'selected_attribute_handles': selected_attribute_handles,
        'matched_candidates': matched_candidates,
        'should_execute': bool(selected_attribute_handles),
        'confidence': 'high' if selected_attribute_handles else 'low',
        'rationale': (
            'Exakter Treffer auf attribute_name.'
            if selected_attribute_handles
            else 'Kein exakter Treffer auf attribute_name.'
        ),
    }

# Auswahl des Attributs aus der Attributliste 
def _select_pf_object_attributes_llm_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
    attribute_listing: dict,
) -> Dict[str, Any]:
    project_name = services['project_name']

    if str((instruction or {}).get('data_source_preference') or 'base').strip().lower() == 'result':
        return _select_pf_object_result_attributes_llm_with_services(
            services=services,
            instruction=instruction,
            resolution=resolution,
            attribute_listing=attribute_listing,
        )

    object_payload = (attribute_listing.get('object', {}) or {}) if isinstance(attribute_listing, dict) else {}
    object_name = object_payload.get('name')
    entity_type = instruction.get('entity_type') if isinstance(instruction, dict) else None
    request_text = (
        instruction.get('attribute_request_text')
        or instruction.get('entity_name_raw')
        or ''
    ) if isinstance(instruction, dict) else ''

    requested_attribute_names = instruction.get('requested_attribute_names', []) if isinstance(instruction, dict) else []
    object_full_name = object_payload.get('full_name')
    pf_object = _get_object_by_full_name(services['app'], object_full_name) if object_full_name else None

    if pf_object is None:
        return {
            'status': 'error',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'attribute_listing': attribute_listing,
            'error': 'pf_object_not_found',
            'details': f'Das aufgelöste Objekt konnte in PowerFactory nicht geladen werden: {object_full_name}',
        }

    # ============================================================
    # SINGLE SOURCE OF TRUTH: PF-EXPORT-LISTE (attribute_name + attribute_description)
    # ============================================================
    export_attribute_options = _build_pf_description_attribute_options(pf_object)
    if not export_attribute_options:
        return {
            'status': 'error',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'attribute_listing': attribute_listing,
            'error': 'empty_attribute_options',
            'details': 'Für das ausgewählte Objekt stehen keine exportierten PowerFactory-Attribute zur Verfügung.',
        }

    available_handles = {
        item.get('handle')
        for item in export_attribute_options
        if item.get('handle')
    }

    candidate_attribute_handles: List[str] = []

    # ============================================================
    # 1) STRICT EXACT MATCH GEGEN attribute_name (PF-EXPORT)
    # ============================================================
    exact_match_result = _match_requested_attribute_names_exact(
        requested_attribute_names=requested_attribute_names,
        attribute_options=export_attribute_options,
    )

    if exact_match_result.get('should_execute'):
        selected_handles = [
            handle for handle in exact_match_result.get('selected_attribute_handles', [])
            if handle in available_handles
        ]
        selected_attributes = [
            item for item in exact_match_result.get('matched_candidates', [])
            if item.get('handle') in selected_handles
        ]

        selection_debug = {
            'project': project_name,
            'entity_type': entity_type,
            'object_name': object_name,
            'request_text': request_text,
            'requested_attribute_names': requested_attribute_names,
            'requested_attribute_name_extraction': instruction.get('requested_attribute_name_extraction'),
            'selection_mode': 'exact_attribute_name',
            'exact_match_result': exact_match_result,
            'pf_description_attribute_count': len(export_attribute_options),
            'final_selected_handles': selected_handles,
            'final_rationale': exact_match_result.get('rationale'),
            'final_should_execute': True,
            'final_confidence': 'high',
        }
        _print_debug_block('Attribute Selection', selection_debug)

        instruction_out = dict(instruction)
        instruction_out['selected_attribute_handles'] = selected_handles
        instruction_out['attribute_selection_debug'] = selection_debug

        return {
            'status': 'ok',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction_out,
            'resolution': resolution,
            'attribute_listing': {
                **(attribute_listing if isinstance(attribute_listing, dict) else {}),
                'attribute_options': export_attribute_options,
            },
            'selected_attribute_handles': selected_handles,
            'selected_attributes': selected_attributes,
            'llm_decision': {
                'path': 'exact_attribute_name',
                'selected_attribute_names': exact_match_result.get('selected_attribute_names', []),
                'confidence': 'high',
                'rationale': exact_match_result.get('rationale'),
                'should_execute': True,
            },
            'selection_debug': selection_debug,
        }

    # ============================================================
    # 2) ZWEISTUFIGER MATCH GEGEN attribute_description (PF-EXPORT)
    # ============================================================
    pf_description_match = _match_pf_attributes_by_description_with_llm(
        obj=pf_object,
        entity_type=entity_type or '',
        object_name=object_name or '',
        user_input=request_text,
        source_preference='base',
    )

    pf_description_attribute_count = len(export_attribute_options)
    pf_description_match_candidates = pf_description_match.get('matched_candidates', []) or []

    pf_description_match_handles: List[str] = []
    seen_handles = set()
    for item in pf_description_match_candidates:
        if not isinstance(item, dict):
            continue

        handle = item.get('handle')
        if not handle:
            attr_name = item.get('attribute_name')
            if attr_name:
                handle = f'attr::{attr_name}'

        if not handle or handle not in available_handles or handle in seen_handles:
            continue

        seen_handles.add(handle)
        pf_description_match_handles.append(handle)

    pf_description_confidence = str(
        (pf_description_match.get('llm_decision') or {}).get('confidence', '')
    ).strip().lower()

    if (
        pf_description_match.get('status') == 'ok'
        and (pf_description_match.get('llm_decision') or {}).get('should_execute')
        and pf_description_confidence == 'high'
        and pf_description_match_handles
    ):
        selected_attributes = [
            item for item in pf_description_match_candidates
            if item.get('handle') in pf_description_match_handles
            or (item.get('attribute_name') and f"attr::{item.get('attribute_name')}" in pf_description_match_handles)
        ]

        selection_debug = {
            'project': project_name,
            'entity_type': entity_type,
            'object_name': object_name,
            'request_text': request_text,
            'requested_attribute_names': requested_attribute_names,
            'requested_attribute_name_extraction': instruction.get('requested_attribute_name_extraction'),
            'selection_mode': 'attribute_description_llm',
            'exact_match_result': exact_match_result,
            'pf_description_attribute_count': pf_description_attribute_count,
            'pf_description_shortlist': (
                (pf_description_match.get('shortlist') or {}).get('llm_decision')
                if isinstance(pf_description_match, dict) else None
            ),
            'pf_description_shortlisted_attribute_names': (
                (pf_description_match.get('shortlist') or {}).get('shortlisted_attribute_names', [])
                if isinstance(pf_description_match, dict) else []
            ),
            'pf_description_match': pf_description_match.get('llm_decision') if isinstance(pf_description_match, dict) else None,
            'pf_description_match_handles': pf_description_match_handles,
            'pf_description_match_candidates': selected_attributes,
            'final_selected_handles': pf_description_match_handles,
            'final_rationale': (pf_description_match.get('llm_decision') or {}).get('rationale'),
            'final_should_execute': True,
            'final_confidence': (pf_description_match.get('llm_decision') or {}).get('confidence', 'high'),
            'pf_description_status': pf_description_match.get('status') if isinstance(pf_description_match, dict) else None,
        }
        _print_debug_block('Attribute Selection', selection_debug)

        instruction_out = dict(instruction)
        instruction_out['selected_attribute_handles'] = pf_description_match_handles
        instruction_out['attribute_selection_debug'] = selection_debug

        return {
            'status': 'ok',
            'tool': 'select_pf_object_attributes_llm',
            'project': project_name,
            'instruction': instruction_out,
            'resolution': resolution,
            'attribute_listing': {
                **(attribute_listing if isinstance(attribute_listing, dict) else {}),
                'attribute_options': export_attribute_options,
            },
            'selected_attribute_handles': pf_description_match_handles,
            'selected_attributes': selected_attributes,
            'llm_decision': {
                'path': 'attribute_description_llm',
                **((pf_description_match.get('llm_decision') or {})),
            },
            'selection_debug': selection_debug,
        }

    # ============================================================
    # 3) KEIN TREFFER
    # ============================================================
    for handle in pf_description_match_handles:
        if handle and handle not in candidate_attribute_handles:
            candidate_attribute_handles.append(handle)

    candidate_attributes = _build_attribute_candidate_suggestions(
        attribute_options=export_attribute_options,
        handles=candidate_attribute_handles,
        max_items=5,
    )

    selection_debug = {
        'project': project_name,
        'entity_type': entity_type,
        'object_name': object_name,
        'request_text': request_text,
        'requested_attribute_names': requested_attribute_names,
        'requested_attribute_name_extraction': instruction.get('requested_attribute_name_extraction'),
        'selection_mode': 'no_match',
        'exact_match_result': exact_match_result,
        'pf_description_attribute_count': pf_description_attribute_count,
        'pf_description_shortlist': (
            (pf_description_match.get('shortlist') or {}).get('llm_decision')
            if isinstance(pf_description_match, dict) else None
        ),
        'pf_description_shortlisted_attribute_names': (
            (pf_description_match.get('shortlist') or {}).get('shortlisted_attribute_names', [])
            if isinstance(pf_description_match, dict) else []
        ),
        'pf_description_match': pf_description_match.get('llm_decision') if isinstance(pf_description_match, dict) else None,
        'pf_description_match_handles': pf_description_match_handles,
        'pf_description_match_candidates': pf_description_match_candidates,
        'candidate_attribute_handles': candidate_attribute_handles,
        'candidate_attributes': candidate_attributes,
        'final_selected_handles': [],
        'final_rationale': 'Weder exakter attribute_name-Treffer noch sicherer attribute_description-Treffer.',
        'final_should_execute': False,
        'final_confidence': 'low',
        'pf_description_status': pf_description_match.get('status') if isinstance(pf_description_match, dict) else None,
        'pf_description_error': pf_description_match.get('error') if isinstance(pf_description_match, dict) else None,
        'pf_description_details': pf_description_match.get('details') if isinstance(pf_description_match, dict) else None,
    }
    _print_debug_block('Attribute Selection', selection_debug)

    return {
        'status': 'error',
        'tool': 'select_pf_object_attributes_llm',
        'project': project_name,
        'instruction': instruction,
        'resolution': resolution,
        'attribute_listing': {
            **(attribute_listing if isinstance(attribute_listing, dict) else {}),
            'attribute_options': export_attribute_options,
        },
        'error': 'attribute_selection_not_safe',
        'details': 'Die Attributauswahl konnte nicht sicher genug aufgelöst werden. Kandidaten wurden zurückgegeben.',
        'candidate_attribute_handles': candidate_attribute_handles,
        'candidate_attributes': candidate_attributes,
        'llm_decision': {
            'path': 'attribute_description_llm',
            **((pf_description_match.get('llm_decision') or {})),
        },
        'missing_context': ['attribute_selection'],
        'selection_debug': selection_debug,
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
            if read_result.get('status') != 'ok':
                read_result = _read_field_with_dynamic_attr_fallback(obj, field_name, meta)
        if read_result.get('status') == 'ok':
            display_value = read_result.get('display_value')
            if display_value is None:
                display_value = read_result.get('numeric_value')
            if display_value is None:
                display_value = read_result.get('raw_value')
            return {
                'status': 'ok',
                'handle': handle,
                'field_name': field_name,
                'label': meta.get('label', field_name),
                'unit': read_result.get('pf_unit') or meta.get('unit') or read_result.get('unit'),
                'unit_source': 'powerfactory' if read_result.get('pf_unit') else ('semantic' if meta.get('unit') else read_result.get('unit_source')),
                'requires_loadflow': bool(meta.get('requires_loadflow', False)),
                'data_source': 'result' if bool(meta.get('requires_loadflow', False)) else 'base',
                'value': display_value,
                'read_debug': read_result,
            }
        return {
            'status': 'error',
            'handle': handle,
            'field_name': field_name,
            'label': meta.get('label', field_name),
            'unit': read_result.get('pf_unit') or meta.get('unit') or read_result.get('unit'),
            'unit_source': 'powerfactory' if read_result.get('pf_unit') else ('semantic' if meta.get('unit') else read_result.get('unit_source')),
            'requires_loadflow': bool(meta.get('requires_loadflow', False)),
            'data_source': 'result' if bool(meta.get('requires_loadflow', False)) else 'base',
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
                'unit': read_result.get('unit'),
                'unit_source': read_result.get('unit_source'),
                'pf_unit': read_result.get('pf_unit'),
                'requires_loadflow': _attribute_name_likely_requires_loadflow(attr_name),
                'data_source': read_result.get('data_source') or _infer_data_source_from_attr_name(attr_name),
                'value': display_value,
                'read_debug': read_result,
            }
        return {
            'status': 'error',
            'handle': handle,
            'attribute_name': attr_name,
            'label': attr_name,
            'unit': read_result.get('unit'),
            'unit_source': read_result.get('unit_source'),
            'pf_unit': read_result.get('pf_unit'),
            'requires_loadflow': _attribute_name_likely_requires_loadflow(attr_name),
            'data_source': read_result.get('data_source') or _infer_data_source_from_attr_name(attr_name),
            'read_debug': read_result,
        }

    return {
        'status': 'error',
        'error': 'unknown_attribute_handle_kind',
        'handle': handle,
    }

# Liest das ausgewählte Attribut vom konkreten PF-Objekt aus 
def _read_pf_object_attributes_with_services(
    services: Dict[str, Any],
    instruction: dict,
    resolution: dict,
) -> Dict[str, Any]:
    app = services['app']
    studycase = services['studycase']
    project_name = services['project_name']

    selected_match = resolution.get('selected_match') if isinstance(resolution, dict) else None
    selected_matches = resolution.get('selected_matches') if isinstance(resolution, dict) else None
    if not isinstance(selected_matches, list) or not selected_matches:
        selected_matches = [selected_match] if isinstance(selected_match, dict) else []

    if not selected_matches:
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

    pf_objects: List[Tuple[Dict[str, Any], Any]] = []
    missing_full_names: List[str] = []
    for match in selected_matches:
        if not isinstance(match, dict):
            continue
        full_name = match.get('full_name')
        pf_object = _get_object_by_full_name(app, full_name)
        if pf_object is None:
            missing_full_names.append(str(full_name))
            continue
        pf_objects.append((match, pf_object))

    if not pf_objects:
        return {
            'status': 'error',
            'tool': 'read_pf_object_attributes',
            'project': project_name,
            'instruction': instruction,
            'resolution': resolution,
            'error': 'pf_object_not_found',
            'details': 'Keines der aufgelösten Objekte konnte in PowerFactory geladen werden.',
            'missing_full_names': missing_full_names,
        }

    requires_loadflow = False
    selected_data_source = 'base'
    for handle in selected_handles:
        if handle.startswith('field::'):
            field_name = handle.split('::', 1)[1]
            meta = _get_available_data_fields(entity_type).get(field_name, {})
            if bool(meta.get('requires_loadflow', False)):
                requires_loadflow = True
                selected_data_source = 'result'
                break
        if handle.startswith('attr::'):
            attr_name = handle.split('::', 1)[1]
            if _attribute_name_likely_requires_loadflow(attr_name):
                requires_loadflow = True
                selected_data_source = 'result'
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

    selection_mode = resolution.get('selection_mode', 'one') if isinstance(resolution, dict) else 'one'

    if len(pf_objects) == 1 and selection_mode != 'all':
        selected_match, pf_object = pf_objects[0]
        full_name = selected_match.get('full_name')

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
                'data_source': read_result.get('data_source') or ('result' if bool(read_result.get('requires_loadflow', False)) else 'base'),
                'field_name': read_result.get('field_name'),
                'attribute_name': read_result.get('attribute_name'),
                'handle': handle,
            }
            values[handle] = read_result.get('value') if read_result.get('status') == 'ok' else None

        debug_payload = {
            'project': project_name,
            'entity_type': entity_type,
            'object': {
                'name': getattr(pf_object, 'loc_name', None),
                'full_name': full_name,
                'pf_class': pf_object.GetClassName() if hasattr(pf_object, 'GetClassName') else None,
            },
            'selected_attribute_handles': selected_handles,
            'selection_debug_summary': {
                'requested_attribute_names': (instruction.get('attribute_selection_debug', {}) or {}).get('requested_attribute_names', []),
                'final_selected_handles': (instruction.get('attribute_selection_debug', {}) or {}).get('final_selected_handles', []),
                'final_rationale': (instruction.get('attribute_selection_debug', {}) or {}).get('final_rationale'),
            },
            'field_reads': field_debug,
        }
        _print_debug_block('Attribute Reads', debug_payload)
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
            'selected_data_source': selected_data_source,
            'selection_notes': list((instruction or {}).get('result_selection_notes', []) or []),
            'debug': {
                'attribute_selection': instruction.get('attribute_selection_debug', {}),
                'field_reads': field_debug,
            },
        }

    values_by_object: Dict[str, Dict[str, Any]] = {}
    field_metadata: Dict[str, Any] = {}
    field_debug_by_object: Dict[str, Any] = {}
    objects_payload: List[Dict[str, Any]] = []

    for match, pf_object in pf_objects:
        object_name = getattr(pf_object, 'loc_name', None) or match.get('name') or match.get('full_name')
        object_full_name = match.get('full_name')
        object_pf_class = pf_object.GetClassName() if hasattr(pf_object, 'GetClassName') else None

        object_values: Dict[str, Any] = {}
        object_field_debug: Dict[str, Any] = {}

        for handle in selected_handles:
            read_result = _read_attribute_handle(pf_object, entity_type, handle)
            object_field_debug[handle] = read_result
            if handle not in field_metadata:
                field_metadata[handle] = {
                    'label': read_result.get('label', handle),
                    'unit': read_result.get('unit'),
                    'requires_loadflow': bool(read_result.get('requires_loadflow', False)),
                    'data_source': read_result.get('data_source') or ('result' if bool(read_result.get('requires_loadflow', False)) else 'base'),
                    'field_name': read_result.get('field_name'),
                    'attribute_name': read_result.get('attribute_name'),
                    'handle': handle,
                }
            object_values[handle] = read_result.get('value') if read_result.get('status') == 'ok' else None

        values_by_object[object_name] = object_values
        field_debug_by_object[object_name] = object_field_debug
        objects_payload.append({
            'name': object_name,
            'full_name': object_full_name,
            'pf_class': object_pf_class,
        })

    first_object_name = next(iter(values_by_object.keys())) if values_by_object else None
    first_values = values_by_object.get(first_object_name, {}) if first_object_name else {}

    debug_payload = {
        'project': project_name,
        'entity_type': entity_type,
        'objects': objects_payload,
        'selected_attribute_handles': selected_handles,
        'selection_debug_summary': {
            'requested_attribute_names': (instruction.get('attribute_selection_debug', {}) or {}).get('requested_attribute_names', []),
            'final_selected_handles': (instruction.get('attribute_selection_debug', {}) or {}).get('final_selected_handles', []),
            'final_rationale': (instruction.get('attribute_selection_debug', {}) or {}).get('final_rationale'),
        },
        'field_reads': field_debug_by_object,
    }
    _print_debug_block('Attribute Reads', debug_payload)
    return {
        'status': 'ok',
        'tool': 'read_pf_object_attributes',
        'project': project_name,
        'studycase': getattr(studycase, 'loc_name', None),
        'instruction': instruction,
        'resolution': resolution,
        'entity_type': entity_type,
        'object': objects_payload[0] if objects_payload else {},
        'objects': objects_payload,
        'data': {
            'entity_type': entity_type,
            'selected_attribute_handles': selected_handles,
            'field_metadata': field_metadata,
            'values': first_values,
            'values_by_object': values_by_object,
        },
        'loadflow': loadflow_info,
        'selected_data_source': selected_data_source,
        'selection_notes': list((instruction or {}).get('result_selection_notes', []) or []),
        'debug': {
            'attribute_selection': instruction.get('attribute_selection_debug', {}),
            'field_reads': field_debug_by_object,
            'missing_full_names': missing_full_names,
            'selection_mode': selection_mode,
        },
    }

# fasst Ergebnis der Attributabfrage zusammen 
def _summarize_pf_object_data_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    project_name = services['project_name']

    data = result_payload.get('data', {}) if isinstance(result_payload, dict) else {}
    values = data.get('values', {}) if isinstance(data, dict) else {}
    values_by_object = data.get('values_by_object', {}) if isinstance(data, dict) else {}
    field_metadata = data.get('field_metadata', {}) if isinstance(data, dict) else {}
    obj = result_payload.get('object', {}) if isinstance(result_payload, dict) else {}
    objects = result_payload.get('objects', []) if isinstance(result_payload, dict) else []
    loadflow = result_payload.get('loadflow', {}) if isinstance(result_payload, dict) else {}
    selection_notes = result_payload.get('selection_notes', []) if isinstance(result_payload, dict) else []
    if not isinstance(selection_notes, list):
        selection_notes = []

    if isinstance(values_by_object, dict) and values_by_object:
        object_parts: List[str] = []
        messages: List[str] = []
        data_sources = set()

        object_metadata_by_name: Dict[str, Dict[str, Any]] = {}
        if isinstance(objects, list):
            for object_item in objects:
                if isinstance(object_item, dict) and object_item.get('name'):
                    object_metadata_by_name[object_item.get('name')] = object_item

        for object_name, object_values in values_by_object.items():
            object_meta = object_metadata_by_name.get(object_name, {})
            object_pf_class = object_meta.get('pf_class') or '<unknown>'
            parts: List[str] = []

            if not isinstance(object_values, dict):
                continue

            for handle, value in object_values.items():
                meta = field_metadata.get(handle, {}) if isinstance(field_metadata, dict) else {}
                label = meta.get('label', handle)
                unit = meta.get('unit')
                data_source = meta.get('data_source', 'base')
                data_sources.add(data_source)
                source_label = 'Basisdaten' if data_source == 'base' else 'Lastflussergebnis'
                if value is None:
                    parts.append(f'{label}: nicht verfügbar ({source_label})')
                    messages.append(f'{label} für {object_name}: nicht verfügbar ({source_label}).')
                elif unit:
                    parts.append(f'{label}: {value} {unit} ({source_label})')
                    messages.append(f'{label} für {object_name}: {value} {unit} ({source_label}).')
                else:
                    parts.append(f'{label}: {value} ({source_label})')
                    messages.append(f'{label} für {object_name}: {value} ({source_label}).')

            if parts:
                object_parts.append(f"'{object_name}' ({object_pf_class}): " + '; '.join(parts))
            else:
                object_parts.append(f"'{object_name}' ({object_pf_class}): keine Daten verfügbar")

        if not object_parts:
            answer = 'Für die ausgewählten Objekte konnten keine Daten gelesen werden.'
        else:
            if data_sources == {'result'}:
                prefix = 'Lastflussergebnisse'
            elif data_sources == {'base'}:
                prefix = 'Basisdaten'
            else:
                prefix = 'Daten'
            answer = f"{prefix} für {len(object_parts)} Objekte: " + ' | '.join(object_parts)

        if loadflow.get('executed'):
            answer += ' Für die angefragten Ergebnisgrößen wurde zuvor ein Lastfluss gerechnet.'
        for note in selection_notes:
            if isinstance(note, str) and note.strip():
                answer += ' ' + note.strip()
                messages.append(note.strip())

        return {
            'status': 'ok',
            'tool': 'summarize_pf_object_data_result',
            'project': project_name,
            'answer': answer,
            'messages': messages,
        }

    object_name = obj.get('name') or obj.get('full_name') or '<unbekannt>'
    pf_class = obj.get('pf_class') or '<unknown>'

    parts: List[str] = []
    messages: List[str] = []
    data_sources = set()
    for handle, value in values.items():
        meta = field_metadata.get(handle, {}) if isinstance(field_metadata, dict) else {}
        label = meta.get('label', handle)
        unit = meta.get('unit')
        data_source = meta.get('data_source', 'base')
        data_sources.add(data_source)
        source_label = 'Basisdaten' if data_source == 'base' else 'Lastflussergebnis'
        if value is None:
            parts.append(f'{label}: nicht verfügbar ({source_label})')
            messages.append(f'{label} für {object_name}: nicht verfügbar ({source_label}).')
        elif unit:
            parts.append(f'{label}: {value} {unit} ({source_label})')
            messages.append(f'{label} für {object_name}: {value} {unit} ({source_label}).')
        else:
            parts.append(f'{label}: {value} ({source_label})')
            messages.append(f'{label} für {object_name}: {value} ({source_label}).')

    if not parts:
        answer = f"Für '{object_name}' konnten keine Daten gelesen werden."
    else:
        if data_sources == {'result'}:
            prefix = 'Lastflussergebnisse'
        elif data_sources == {'base'}:
            prefix = 'Basisdaten'
        else:
            prefix = 'Daten'
        answer = f"{prefix} für '{object_name}' ({pf_class}): " + '; '.join(parts)
    if loadflow.get('executed'):
        answer += ' Für die angefragten Ergebnisgrößen wurde zuvor ein Lastfluss gerechnet.'
    for note in selection_notes:
        if isinstance(note, str) and note.strip():
            answer += ' ' + note.strip()
            messages.append(note.strip())

    return {
        'status': 'ok',
        'tool': 'summarize_pf_object_data_result',
        'project': project_name,
        'answer': answer,
        'messages': messages,
    }


# Entscheidung, ob Attribut eher base oder result erfordert 
def _classify_pf_object_data_source_with_services(
    services: Dict[str, Any],
    instruction: dict,
) -> Dict[str, Any]:
    project_name = services['project_name']
    request_text = ''
    if isinstance(instruction, dict):
        request_text = str(instruction.get('attribute_request_text') or instruction.get('entity_name_raw') or instruction.get('request_text') or '')
    decision = _resolve_data_source_preference(request_text)
    instruction_out = dict(instruction or {})
    instruction_out['data_source_preference'] = decision.get('effective_data_source', 'base')
    instruction_out['data_source_note'] = decision.get('data_source_note')
    instruction_out['data_source_decision'] = decision
    return {
        'status': 'ok',
        'tool': 'classify_data_source',
        'project': project_name,
        'instruction': instruction_out,
        'selected_data_source': decision.get('selected_data_source'),
        'effective_data_source': decision.get('effective_data_source', 'base'),
        'data_source_note': decision.get('data_source_note'),
        'data_source_decision': decision,
    }



# ------------------------------------------------------------------
# REGISTRY-UNIFIED SUMMARY / CONTROL HELPERS
# ------------------------------------------------------------------
def _summarize_load_catalog_with_services(
    services: Dict[str, Any],
    catalog_result: Dict[str, Any],
) -> Dict[str, Any]:
    loads = catalog_result.get('loads', []) if isinstance(catalog_result, dict) else []
    names = [entry.get('loc_name') for entry in loads if isinstance(entry, dict) and entry.get('loc_name')]
    preview = names[:10]

    if not names:
        answer = 'Im aktiven PowerFactory-Projekt wurden keine Lasten gefunden.'
    elif len(names) <= 10:
        answer = 'Verfügbare Lasten im aktiven PowerFactory-Projekt: ' + ', '.join(names)
    else:
        answer = f"Im aktiven PowerFactory-Projekt wurden {len(names)} Lasten gefunden. Beispiele: " + ', '.join(preview)

    return {
        'status': 'ok',
        'tool': 'summarize_load_catalog',
        'answer': answer,
        'count': len(names),
        'loads': loads,
    }


def _summarize_topology_result_with_services(
    services: Dict[str, Any],
    topology_result: Dict[str, Any],
    graph_result: Dict[str, Any] | None = None,
    inventory_result: Dict[str, Any] | None = None,
    entity_instruction: Dict[str, Any] | None = None,
    entity_resolution: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not topology_result or topology_result.get('status') != 'ok':
        return {
            'status': 'error',
            'tool': 'summarize_topology_result',
            'error': 'missing_topology_result',
            'answer': 'Es liegt kein gültiges Topologieergebnis zur Zusammenfassung vor.',
        }

    selected = topology_result.get('selected_node', {}) or {}
    neighbors = topology_result.get('neighbors', []) or []
    selected_name = selected.get('name') or selected.get('full_name') or '<unbekannt>'
    selected_class = selected.get('pf_class') or '<unknown>'
    selected_type = selected.get('inventory_type') or '<unknown>'
    neighbor_count = topology_result.get('neighbor_count', len(neighbors))

    if neighbor_count == 0:
        answer = f"Für das Asset '{selected_name}' ({selected_class}) wurden im aktuellen PowerFactory-Topologiegraphen keine direkten Nachbarn gefunden."
    else:
        preview_items = []
        for neighbor in neighbors[:10]:
            neighbor_name = neighbor.get('name') or neighbor.get('full_name') or '<unbekannt>'
            neighbor_class = neighbor.get('pf_class') or '<unknown>'
            preview_items.append(f"{neighbor_name} ({neighbor_class})")

        if neighbor_count <= 10:
            answer = f"Direkte Nachbarn von '{selected_name}' ({selected_class}, Typ {selected_type}) im PowerFactory-Topologiegraphen: " + ', '.join(preview_items)
        else:
            answer = f"Für '{selected_name}' ({selected_class}, Typ {selected_type}) wurden {neighbor_count} direkte Nachbarn im PowerFactory-Topologiegraphen gefunden. Beispiele: " + ', '.join(preview_items)

    return {
        'status': 'ok',
        'tool': 'summarize_topology_result',
        'answer': answer,
        'selected_node': selected,
        'neighbor_count': neighbor_count,
        'neighbors': neighbors,
        'graph_summary': graph_result.get('graph_summary', {}) if isinstance(graph_result, dict) else {},
        'inventory_types': inventory_result.get('inventory', {}).get('available_types', []) if isinstance(inventory_result, dict) else [],
        'instruction': entity_instruction,
        'resolution': entity_resolution,
    }


def _build_unsupported_result_with_services(
    services: Dict[str, Any],
    user_input: str,
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    missing_context = classification.get('missing_context', []) if isinstance(classification, dict) else []
    missing_text = ''
    if missing_context:
        missing_text = ' Fehlender Kontext: ' + ', '.join(missing_context) + '.'

    return {
        'status': 'error',
        'tool': 'powerfactory',
        'agent': 'PowerFactoryDomainAgent',
        'error': 'unsupported_powerfactory_request',
        'answer': 'Die Anfrage wurde nach PowerFactory geroutet, passt aber aktuell zu keinem unterstützten PowerFactory-Ablauf oder ist noch nicht sicher ausführbar.' + missing_text,
        'user_input': user_input,
        'classification': classification,
    }
