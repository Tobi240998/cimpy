from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.cimpy_time_analysis.load_cim_data import (
    scan_snapshot_inventory as _scan_snapshot_inventory_raw,
    load_base_snapshot,
    build_network_index_from_snapshot,
    load_snapshots_for_time_window,
    load_cim_snapshots,
)
from cimpy.cimpy_time_analysis.cim_snapshot_cache import (
    preprocess_snapshots_for_states,
    summarize_snapshot_cache,
)
from cimpy.cimpy_time_analysis.llm_object_mapping import (
    interpret_user_query,
    interpret_equipment_type_query,
    resolve_requested_base_attributes,
)
from cimpy.cimpy_time_analysis.llm_cim_orchestrator import handle_user_query
from cimpy.cimpy_time_analysis.langchain_llm import get_llm


_EXCLUDED_EQUIPMENT_CLASS_NAMES = {
    "Terminal",
    "ConnectivityNode",
    "TopologicalNode",
    "BaseVoltage",
    "Substation",
    "VoltageLevel",
    "Bay",
    "GeographicalRegion",
    "SubGeographicalRegion",
    "Location",
    "CoordinateSystem",
    "PositionPoint",
}


class EquipmentTypeDecision(BaseModel):
    selected_type: Optional[str] = Field(
        default=None,
        description="Exact class name from the provided equipment type list, or null if no safe match exists.",
    )
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the selection")
    should_execute: bool = Field(
        description="True only if the selected type is a safe unambiguous choice."
    )
    alternatives: List[str] = Field(default_factory=list)


class EquipmentInstanceDecision(BaseModel):
    selected_equipment_id: Optional[str] = Field(
        default=None,
        description="Exact canonical_id from the provided candidate list, or null if no safe match exists.",
    )
    confidence: str = Field(description="One of: high, medium, low")
    rationale: str = Field(description="Short explanation for the selection")
    should_execute: bool = Field(
        description="True only if the selected equipment candidate is a safe unambiguous choice."
    )
    alternatives: List[str] = Field(default_factory=list)


class ParsedQueryNormalizationDecision(BaseModel):
    equipment_type_hint: Optional[str] = Field(default=None)
    equipment_name_hint: Optional[str] = Field(default=None)


def _normalize_cim_root(cim_root: Optional[str] = None) -> str:
    return cim_root or CIM_ROOT


# ------------------------------------------------------------------
# CONTEXT / SERVICES
# ------------------------------------------------------------------
def build_cim_services(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a minimal CIM service/context payload.

    Kept intentionally lightweight so the current logic stays unchanged while
    the module becomes structurally closer to the PowerFactory MCP layer.
    """
    normalized_root = _normalize_cim_root(cim_root)
    return {
        "status": "ok",
        "tool": "cim_context",
        "cim_root": normalized_root,
    }


# ------------------------------------------------------------------
# GENERIC CIM OBJECT / EQUIPMENT HELPERS
# ------------------------------------------------------------------
def _collect_all_cim_objects(container):
    objects = []

    if isinstance(container, dict):
        for value in container.values():
            objects.extend(_collect_all_cim_objects(value))

    elif isinstance(container, list):
        for value in container:
            objects.extend(_collect_all_cim_objects(value))

    else:
        cls = container.__class__
        if cls.__module__.startswith("cimpy"):
            objects.append(container)

    return objects



def _canonical_cim_id(value):
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
    return s.lower() if s else None



def _safe_name(value):
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None



def _safe_description(value):
    if value is None:
        return None
    for attr in ("description", "desc", "shortName"):
        text = getattr(value, attr, None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None



def _is_equipment_candidate(obj: Any) -> bool:
    if obj is None:
        return False

    class_name = obj.__class__.__name__
    if not class_name:
        return False

    mrid = getattr(obj, "mRID", None)
    if not isinstance(mrid, str) or not mrid.strip():
        return False

    if class_name.startswith("Sv"):
        return False

    if class_name in _EXCLUDED_EQUIPMENT_CLASS_NAMES:
        return False

    return True



def _build_equipment_catalog(container) -> Dict[str, Any]:
    all_objects = _collect_all_cim_objects(container)

    equipment_by_type: Dict[str, List[Any]] = {}
    equipment_by_mrid: Dict[str, Any] = {}
    equipment_name_index_all: Dict[str, Dict[str, Any]] = {}
    equipment_catalog: List[Dict[str, Any]] = []
    equipment_type_counts: Dict[str, int] = {}

    seen_ids = set()

    for obj in all_objects:
        if not _is_equipment_candidate(obj):
            continue

        class_name = obj.__class__.__name__
        canonical_id = _canonical_cim_id(obj)
        if not canonical_id or canonical_id in seen_ids:
            continue

        seen_ids.add(canonical_id)
        equipment_by_type.setdefault(class_name, []).append(obj)
        equipment_by_mrid[canonical_id] = obj
        equipment_type_counts[class_name] = equipment_type_counts.get(class_name, 0) + 1

        name = _safe_name(obj)
        if name:
            equipment_name_index_all.setdefault(class_name, {})[name.strip().lower()] = obj

        equipment_catalog.append({
            "class_name": class_name,
            "name": name,
            "mRID": getattr(obj, "mRID", None),
            "canonical_id": canonical_id,
            "description": _safe_description(obj),
        })

    for class_name, items in equipment_by_type.items():
        items.sort(key=lambda obj: ((_safe_name(obj) or "").lower(), _canonical_cim_id(obj) or ""))

    equipment_catalog.sort(
        key=lambda item: (
            item.get("class_name") or "",
            (item.get("name") or "").lower(),
            item.get("canonical_id") or "",
        )
    )

    return {
        "equipment_types": sorted(equipment_by_type.keys()),
        "equipment_type_counts": equipment_type_counts,
        "equipment_by_type": equipment_by_type,
        "equipment_by_mrid": equipment_by_mrid,
        "equipment_name_index_all": equipment_name_index_all,
        "equipment_catalog": equipment_catalog,
        "num_equipment_objects": len(equipment_catalog),
    }



def _merge_equipment_catalog_into_network_index(network_index: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(network_index or {})
    merged.update({
        "equipment_types": catalog.get("equipment_types", []),
        "equipment_type_counts": catalog.get("equipment_type_counts", {}),
        "equipment_by_type": catalog.get("equipment_by_type", {}),
        "equipment_by_mrid": catalog.get("equipment_by_mrid", {}),
        "equipment_name_index_all": catalog.get("equipment_name_index_all", {}),
        "equipment_catalog": catalog.get("equipment_catalog", []),
        "num_equipment_objects": catalog.get("num_equipment_objects", 0),
    })
    return merged


# ------------------------------------------------------------------
# LLM-BASED EQUIPMENT RESOLUTION
# ------------------------------------------------------------------
def _build_equipment_type_match_chain():
    parser = PydanticOutputParser(pydantic_object=EquipmentTypeDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You match a user's requested CIM equipment reference to an existing CIM equipment type.\n"
            "You may only select a type name that appears exactly in the provided candidate list.\n"
            "Do not invent type names.\n"
            "If there is no safe unambiguous match, return selected_type=null and should_execute=false.\n"
            "Use high confidence only for a clearly dominant match.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Available equipment types:\n{equipment_types}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser



def _build_equipment_instance_match_chain():
    parser = PydanticOutputParser(pydantic_object=EquipmentInstanceDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You match a user's requested CIM equipment reference to an existing equipment candidate.\n"
            "You may only select a canonical_id that appears exactly in the provided candidate list.\n"
            "Do not invent IDs or names.\n"
            "If there is no safe unambiguous match, return selected_equipment_id=null and should_execute=false.\n"
            "Use high confidence only for a clearly dominant match.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Selected equipment type:\n{selected_type}\n\n"
            "Available equipment candidates:\n{equipment_candidates}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser



def _build_query_normalization_chain():
    parser = PydanticOutputParser(pydantic_object=ParsedQueryNormalizationDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract the most likely CIM equipment type hint and equipment name hint from the user request.\n"
            "Return short strings only.\n"
            "If a value is not identifiable, return null.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser



def _normalize_hint_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()



def _select_equipment_type_with_llm(user_input: str, equipment_types: List[str], equipment_type_counts: Dict[str, int]) -> Dict[str, Any]:
    if not equipment_types:
        return {
            "status": "error",
            "error": "no_equipment_types",
            "details": "Es wurden keine Equipment-Typen im Netzwerkindex gefunden.",
        }

    if len(equipment_types) == 1:
        selected_type = equipment_types[0]
        return {
            "status": "ok",
            "selected_type": selected_type,
            "llm_decision": {
                "selected_type": selected_type,
                "confidence": "high",
                "rationale": "Only one equipment type is available.",
                "should_execute": True,
                "alternatives": [],
            },
        }

    try:
        chain, parser = _build_equipment_type_match_chain()
        decision = chain.invoke({
            "user_input": user_input,
            "equipment_types": "\n".join(
                f"- {type_name} (count={equipment_type_counts.get(type_name, 0)})"
                for type_name in equipment_types
            ),
            "format_instructions": parser.get_format_instructions(),
        })
    except Exception as e:
        return {
            "status": "error",
            "error": "llm_equipment_type_match_failed",
            "details": str(e),
        }

    selected_type = decision.selected_type
    if selected_type not in equipment_types:
        return {
            "status": "error",
            "error": "invalid_equipment_type_selection",
            "details": "Das LLM hat keinen gültigen exakten Equipment-Typ aus der Kandidatenliste zurückgegeben.",
            "llm_decision": decision.model_dump(),
            "equipment_types": equipment_types,
        }

    if not decision.should_execute or decision.confidence.lower() == "low":
        return {
            "status": "error",
            "error": "equipment_type_match_not_safe",
            "details": "Das LLM hat keinen ausreichend sicheren Equipment-Typ gefunden.",
            "llm_decision": decision.model_dump(),
            "equipment_types": equipment_types,
        }

    return {
        "status": "ok",
        "selected_type": selected_type,
        "llm_decision": decision.model_dump(),
    }



def _select_equipment_instance_with_llm(user_input: str, selected_type: str, equipment_candidates: List[Any]) -> Dict[str, Any]:
    candidate_rows: List[Dict[str, Any]] = []
    candidate_ids: List[str] = []

    for obj in equipment_candidates or []:
        canonical_id = _canonical_cim_id(obj)
        if not canonical_id:
            continue
        row = {
            "canonical_id": canonical_id,
            "name": _safe_name(obj),
            "mRID": getattr(obj, "mRID", None),
            "description": _safe_description(obj),
            "class_name": obj.__class__.__name__,
        }
        candidate_rows.append(row)
        candidate_ids.append(canonical_id)

    if not candidate_rows:
        return {
            "status": "error",
            "error": "no_equipment_candidates_for_type",
            "details": f"Für den Equipment-Typ '{selected_type}' wurden keine Kandidaten gefunden.",
        }

    if len(candidate_rows) == 1:
        row = candidate_rows[0]
        return {
            "status": "ok",
            "selected_equipment_id": row["canonical_id"],
            "selected_match": row,
            "llm_decision": {
                "selected_equipment_id": row["canonical_id"],
                "confidence": "high",
                "rationale": "Only one equipment candidate is available for the selected type.",
                "should_execute": True,
                "alternatives": [],
            },
        }

    candidate_text = "\n".join(
        "- canonical_id={canonical_id} | name={name} | mRID={mrid} | description={description}".format(
            canonical_id=row.get("canonical_id"),
            name=row.get("name"),
            mrid=row.get("mRID"),
            description=row.get("description"),
        )
        for row in candidate_rows
    )

    try:
        chain, parser = _build_equipment_instance_match_chain()
        decision = chain.invoke({
            "user_input": user_input,
            "selected_type": selected_type,
            "equipment_candidates": candidate_text,
            "format_instructions": parser.get_format_instructions(),
        })
    except Exception as e:
        return {
            "status": "error",
            "error": "llm_equipment_instance_match_failed",
            "details": str(e),
        }

    selected_equipment_id = decision.selected_equipment_id
    if selected_equipment_id not in candidate_ids:
        return {
            "status": "error",
            "error": "invalid_equipment_instance_selection",
            "details": "Das LLM hat keine gültige canonical_id aus der Kandidatenliste zurückgegeben.",
            "llm_decision": decision.model_dump(),
        }

    if not decision.should_execute or decision.confidence.lower() == "low":
        return {
            "status": "error",
            "error": "equipment_instance_match_not_safe",
            "details": "Das LLM hat kein ausreichend sicheres konkretes Equipment gefunden.",
            "llm_decision": decision.model_dump(),
        }

    selected_match = next(row for row in candidate_rows if row["canonical_id"] == selected_equipment_id)

    return {
        "status": "ok",
        "selected_equipment_id": selected_equipment_id,
        "selected_match": selected_match,
        "llm_decision": decision.model_dump(),
    }



def _normalize_user_query_hints(user_input: str) -> Dict[str, Any]:
    try:
        chain, parser = _build_query_normalization_chain()
        decision = chain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions(),
        })
        return {
            "status": "ok",
            "equipment_type_hint": decision.equipment_type_hint,
            "equipment_name_hint": decision.equipment_name_hint,
        }
    except Exception as e:
        return {
            "status": "error",
            "equipment_type_hint": None,
            "equipment_name_hint": None,
            "details": str(e),
        }



def _resolve_equipment_via_catalog(user_input: str, network_index: Dict[str, Any] | None) -> Dict[str, Any]:
    network_index = network_index or {}
    equipment_types = network_index.get("equipment_types", []) or []
    equipment_type_counts = network_index.get("equipment_type_counts", {}) or {}
    equipment_by_type = network_index.get("equipment_by_type", {}) or {}
    equipment_by_mrid = network_index.get("equipment_by_mrid", {}) or {}

    if not equipment_types or not equipment_by_type:
        return {
            "status": "error",
            "error": "missing_equipment_catalog",
            "details": "Im Netzwerkindex ist kein Equipment-Katalog vorhanden.",
        }

    query_hint_result = _normalize_user_query_hints(user_input)
    equipment_type_hint = _normalize_hint_text(query_hint_result.get("equipment_type_hint"))

    typed_candidates = equipment_types
    if equipment_type_hint:
        hinted = [t for t in equipment_types if equipment_type_hint in t.lower()]
        if hinted:
            typed_candidates = hinted

    type_result = _select_equipment_type_with_llm(
        user_input=user_input,
        equipment_types=typed_candidates,
        equipment_type_counts=equipment_type_counts,
    )
    if type_result.get("status") != "ok":
        return type_result

    selected_type = type_result["selected_type"]
    instance_result = _select_equipment_instance_with_llm(
        user_input=user_input,
        selected_type=selected_type,
        equipment_candidates=equipment_by_type.get(selected_type, []),
    )
    if instance_result.get("status") != "ok":
        return {
            **instance_result,
            "selected_type": selected_type,
            "type_llm_decision": type_result.get("llm_decision"),
        }

    selected_equipment_id = instance_result["selected_equipment_id"]
    resolved_object = equipment_by_mrid.get(selected_equipment_id)

    if resolved_object is None:
        return {
            "status": "error",
            "error": "selected_equipment_not_found_in_catalog",
            "details": f"Das ausgewählte Equipment konnte nicht über canonical_id geladen werden: {selected_equipment_id}",
            "selected_type": selected_type,
            "selected_equipment_id": selected_equipment_id,
        }

    return {
        "status": "ok",
        "selected_type": selected_type,
        "selected_equipment_id": selected_equipment_id,
        "selected_match": instance_result.get("selected_match"),
        "resolved_object": resolved_object,
        "type_llm_decision": type_result.get("llm_decision"),
        "instance_llm_decision": instance_result.get("llm_decision"),
        "query_hint_result": query_hint_result,
    }


# ------------------------------------------------------------------
# INTERNAL HELPERS (moved from registry, logic unchanged)
# ------------------------------------------------------------------
def _extract_required_state_types(parsed_query: Dict[str, Any] | None) -> List[str]:
    parsed_query = parsed_query or {}
    state_detected = parsed_query.get("state_detected", []) or []

    state_types: List[str] = []

    if "SvVoltage" in state_detected:
        state_types.append("SvVoltage")
    if "SvPowerFlow" in state_detected:
        state_types.append("SvPowerFlow")

    out: List[str] = []
    seen = set()

    for state_type in state_types:
        if state_type not in seen:
            out.append(state_type)
            seen.add(state_type)

    return out



def _should_load_states(parsed_query: Dict[str, Any] | None) -> bool:
    return len(_extract_required_state_types(parsed_query)) > 0


# ------------------------------------------------------------------
# LOW-LEVEL TOOL IMPLEMENTATIONS
# ------------------------------------------------------------------
def _scan_snapshot_inventory_with_services(
    services: Dict[str, Any],
) -> Dict[str, Any]:
    cim_root = services["cim_root"]

    inventory = _scan_snapshot_inventory_raw(cim_root)

    base_snapshot = load_base_snapshot(
        root_folder=cim_root,
        snapshot_inventory=inventory,
    )

    network_index = build_network_index_from_snapshot(base_snapshot)
    equipment_catalog = _build_equipment_catalog(base_snapshot)
    network_index = _merge_equipment_catalog_into_network_index(
        network_index=network_index,
        catalog=equipment_catalog,
    )

    if not network_index or not network_index.get("equipment_name_index"):
        return {
            "status": "error",
            "tool": "scan_snapshot_inventory",
            "error": "invalid_network_index",
            "answer": "Es konnte kein gültiger Netzwerkindex aus dem Basissnapshot aufgebaut werden.",
        }

    return {
        "status": "ok",
        "tool": "scan_snapshot_inventory",
        "cim_root": cim_root,
        "snapshot_inventory": inventory,
        "base_snapshot": base_snapshot,
        "network_index": network_index,
        "equipment_catalog_summary": {
            "equipment_types": equipment_catalog.get("equipment_types", []),
            "equipment_type_counts": equipment_catalog.get("equipment_type_counts", {}),
            "num_equipment_objects": equipment_catalog.get("num_equipment_objects", 0),
        },
    }



def _resolve_cim_object_with_services(
    services: Dict[str, Any],
    user_input: str,
    network_index: Dict[str, Any] | None,
) -> Dict[str, Any]:
    parsed_query = interpret_user_query(
        user_input=user_input,
        network_index=network_index,
    )
    if not isinstance(parsed_query, dict):
        parsed_query = {}

    network_index = network_index or {}
    equipment_name_index = network_index.get("equipment_name_index", {}) or {}
    equipment_by_mrid = network_index.get("equipment_by_mrid", {}) or {}

    equipment_selection = parsed_query.get("equipment_selection", []) or []
    resolved = None

    if equipment_selection:
        first_sel = equipment_selection[0]
        if isinstance(first_sel, dict):
            eq_type = first_sel.get("equipment_type")
            eq_key = first_sel.get("equipment_key")
            eq_id = first_sel.get("equipment_id")
            if eq_type and eq_key:
                resolved = equipment_name_index.get(eq_type, {}).get(eq_key)
            if resolved is None and eq_id:
                resolved = equipment_by_mrid.get(_canonical_cim_id(eq_id))
        else:
            resolved = first_sel

    resolution_mode = "interpret_user_query"
    equipment_resolution_debug: Dict[str, Any] = {
        "parsed_query_equipment_selection": equipment_selection,
    }

    if resolved is None:
        catalog_resolution = _resolve_equipment_via_catalog(
            user_input=user_input,
            network_index=network_index,
        )

        equipment_resolution_debug = catalog_resolution

        if catalog_resolution.get("status") == "ok":
            resolved = catalog_resolution.get("resolved_object")
            parsed_query["resolved_equipment_type"] = catalog_resolution.get("selected_type")
            parsed_query["resolved_equipment_id"] = catalog_resolution.get("selected_equipment_id")
            parsed_query["equipment_resolution_strategy"] = "catalog_type_then_instance_llm"
            resolution_mode = "catalog_type_then_instance_llm"

    return {
        "status": "ok",
        "tool": "resolve_cim_object",
        "cim_root": services["cim_root"],
        "user_input": user_input,
        "parsed_query": parsed_query,
        "resolved_object": resolved,
        "equipment_obj": resolved,
        "resolution_mode": resolution_mode,
        "equipment_resolution_debug": equipment_resolution_debug,
    }


def _list_equipment_of_type_with_services(
    services: Dict[str, Any],
    user_input: str,
    network_index: Dict[str, Any] | None,
) -> Dict[str, Any]:
    network_index = network_index or {}
    equipment_types = network_index.get("equipment_types", []) or []
    equipment_type_counts = network_index.get("equipment_type_counts", {}) or {}
    equipment_by_type = network_index.get("equipment_by_type", {}) or {}

    if not equipment_types or not equipment_by_type:
        return {
            "status": "error",
            "tool": "list_equipment_of_type",
            "cim_root": services["cim_root"],
            "error": "missing_equipment_catalog",
            "answer": "Im Netzwerkindex ist kein Equipment-Katalog vorhanden.",
        }

    type_query = interpret_equipment_type_query(
        user_input=user_input,
        network_index=network_index,
        allowed_equipment_types=equipment_types,
    )
    selected_type = type_query.get("selected_type")

    if not selected_type or selected_type not in equipment_by_type:
        query_hint_result = _normalize_user_query_hints(user_input)
        equipment_type_hint = _normalize_hint_text(query_hint_result.get("equipment_type_hint"))

        typed_candidates = equipment_types
        if equipment_type_hint:
            hinted = [t for t in equipment_types if equipment_type_hint in t.lower()]
            if hinted:
                typed_candidates = hinted

        type_result = _select_equipment_type_with_llm(
            user_input=user_input,
            equipment_types=typed_candidates,
            equipment_type_counts=equipment_type_counts,
        )
        if type_result.get("status") != "ok":
            return {
                "status": "error",
                "tool": "list_equipment_of_type",
                "cim_root": services["cim_root"],
                "error": type_result.get("error", "equipment_type_selection_failed"),
                "answer": "Der angefragte Equipment-Typ konnte nicht sicher aufgelöst werden.",
                "equipment_resolution_debug": {
                    **type_result,
                    "query_hint_result": query_hint_result,
                    "type_query": type_query,
                },
            }
        selected_type = type_result["selected_type"]
        resolution_debug = {
            "selected_type": selected_type,
            "type_llm_decision": type_result.get("llm_decision"),
            "query_hint_result": query_hint_result,
            "type_query": type_query,
            "resolution_strategy": "type_only_query_then_type_fallback_llm",
        }
    else:
        resolution_debug = {
            "selected_type": selected_type,
            "type_query": type_query,
            "resolution_strategy": "type_only_query",
        }

    equipment_objects = equipment_by_type.get(selected_type, []) or []

    equipment_items: List[Dict[str, Any]] = []
    for obj in equipment_objects:
        equipment_items.append({
            "canonical_id": _canonical_cim_id(obj),
            "name": _safe_name(obj),
            "mRID": getattr(obj, "mRID", None),
            "description": _safe_description(obj),
            "class_name": obj.__class__.__name__,
        })

    equipment_items.sort(
        key=lambda item: (
            (item.get("name") or "").lower(),
            item.get("canonical_id") or "",
        )
    )

    preview_items = equipment_items[:25]
    preview_names = [item.get("name") or item.get("canonical_id") for item in preview_items]
    answer = (
        f"Ich habe {len(equipment_items)} Objekte vom Typ '{selected_type}' gefunden: "
        + ", ".join(preview_names)
    )
    if len(equipment_items) > len(preview_items):
        answer += f" … und {len(equipment_items) - len(preview_items)} weitere."

    return {
        "status": "ok",
        "tool": "list_equipment_of_type",
        "cim_root": services["cim_root"],
        "selected_type": selected_type,
        "equipment_items": equipment_items,
        "equipment_count": len(equipment_items),
        "answer": answer,
        "equipment_resolution_debug": resolution_debug,
    }







_CGMES_BASE_ATTRIBUTE_UNITS: Dict[str, str | None] = {
    # CGMES exchange multiplier rule:
    # - k for volt
    # - M for W, VA, VAr
    # - 1 for all remaining UnitSymbol values
    # Non-quantitative attributes intentionally map to None.
    "name": None,
    "mRID": None,
    "description": None,
    "type": None,
    "operatingMode": None,
    "connectionKind": None,
    "grounded": None,
    "endNumber": None,
    "phaseAngleClock": None,
    "ratedU": "kV",
    "maxU": "kV",
    "minU": "kV",
    "ratedS": "MVA",
    "p": "MW",
    "initialP": "MW",
    "nominalP": "MW",
    "maxOperatingP": "MW",
    "minOperatingP": "MW",
    "q": "MVAr",
    "baseQ": "MVAr",
    "maxQ": "MVAr",
    "minQ": "MVAr",
    "r": "ohm",
    "r0": "ohm",
    "r2": "ohm",
    "rground": "ohm",
    "x": "ohm",
    "x0": "ohm",
    "x2": "ohm",
    "xground": "ohm",
    "b": "S",
    "b0": "S",
    "g": "S",
    "g0": "S",
}


def _get_cgmes_unit_for_base_attribute(attr_name: str) -> str | None:
    if not attr_name:
        return None
    return _CGMES_BASE_ATTRIBUTE_UNITS.get(str(attr_name).strip())


def _format_base_value_for_answer(value: Any, unit: str | None) -> str:
    if isinstance(value, list):
        rendered_rows = []
        for row in value:
            if isinstance(row, dict):
                row_value = row.get("value")
                row_unit = row.get("unit") or unit
                suffix = f" {row_unit}" if row_unit else ""
                rendered_rows.append(
                    "{end_label}{value}{suffix}".format(
                        end_label=(f"Ende {row.get('endNumber')}: " if row.get('endNumber') is not None else ""),
                        value=row_value,
                        suffix=suffix,
                    )
                )
            else:
                suffix = f" {unit}" if unit else ""
                rendered_rows.append(f"{row!r}{suffix}")
        return "[" + ", ".join(rendered_rows) + "]"

    suffix = f" {unit}" if unit else ""
    return f"{value!r}{suffix}"


def _collect_base_attribute_value(resolved_object: Any, attr_name: str) -> Any:
    if resolved_object is None or not attr_name:
        return None

    class_name = resolved_object.__class__.__name__

    if class_name == "PowerTransformer":
        ends = getattr(resolved_object, "PowerTransformerEnd", None) or []
        end_rows = []
        for end in ends:
            if not hasattr(end, attr_name):
                continue
            value = getattr(end, attr_name, None)
            if value is None:
                continue
            end_rows.append({
                "endNumber": getattr(end, "endNumber", None),
                "terminal_id": _canonical_cim_id(getattr(end, "Terminal", None)),
                "value": value,
                "unit": _get_cgmes_unit_for_base_attribute(attr_name),
            })
        if end_rows:
            end_rows.sort(key=lambda row: ((row.get("endNumber") is None), row.get("endNumber"), row.get("terminal_id") or ""))
            return end_rows

    if class_name == "SynchronousMachine":
        generating_unit = getattr(resolved_object, "GeneratingUnit", None)
        if generating_unit is not None and hasattr(generating_unit, attr_name):
            value = getattr(generating_unit, attr_name, None)
            if value is not None:
                return value

    return getattr(resolved_object, attr_name, None)

def _read_cim_base_values_with_services(
    services: Dict[str, Any],
    user_input: str,
    resolved_object: Any,
    parsed_query: Dict[str, Any] | None = None,
    analysis_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parsed_query = parsed_query or {}

    if analysis_plan is not None:
        print(f"analysis_plan: {analysis_plan}")
    print(f"equipment_detected: {parsed_query.get('equipment_detected', [])}")
    print(f"state_detected: {parsed_query.get('state_detected', [])}")
    print(f"metric: {parsed_query.get('metric')}")
    print(f"time_label: {parsed_query.get('time_label')}")

    if resolved_object is None:
        print("equipment_obj: None")
        return {
            "status": "error",
            "tool": "read_cim_base_values",
            "cim_root": services["cim_root"],
            "error": "missing_resolved_object",
            "answer": "Es wurde kein aufgelöstes CIM-Equipment für die Basiswertabfrage gefunden.",
            "base_attribute_debug": {
                "resolution_mode": "missing_resolved_object",
                "available_attributes": [],
                "selected_attributes": [],
            },
        }

    print(f"equipment_obj type: {type(resolved_object)}")
    print(f"equipment_obj name: {_safe_name(resolved_object)}")
    print(f"equipment_obj id: {_canonical_cim_id(resolved_object)} {getattr(resolved_object, 'mRID', None)}")

    selection_result = resolve_requested_base_attributes(
        user_input=user_input,
        equipment_obj=resolved_object,
    )

    print(f"base_attribute_debug: {selection_result}")

    selected_attributes = selection_result.get("selected_attributes", []) or []
    if not selected_attributes:
        return {
            "status": "error",
            "tool": "read_cim_base_values",
            "cim_root": services["cim_root"],
            "error": "no_matching_base_attributes",
            "answer": "Für die Anfrage konnten keine passenden technischen Basisattribute gefunden werden.",
            "base_attribute_debug": selection_result,
        }

    values: Dict[str, Any] = {}
    units: Dict[str, str | None] = {}
    for attr_name in selected_attributes:
        values[attr_name] = _collect_base_attribute_value(resolved_object, attr_name)
        units[attr_name] = _get_cgmes_unit_for_base_attribute(attr_name)

    print(f"selected_attributes: {selected_attributes}")
    print(f"base_values: {values}")
    print(f"base_value_units: {units}")

    equipment_name = _safe_name(resolved_object) or getattr(resolved_object, "mRID", None) or "<unbekannt>"
    parts = [
        f"{attr}={_format_base_value_for_answer(values.get(attr), units.get(attr))}"
        for attr in selected_attributes
    ]
    answer = f"Basiswerte für {equipment_name}: " + ", ".join(parts)

    return {
        "status": "ok",
        "tool": "read_cim_base_values",
        "cim_root": services["cim_root"],
        "selected_attributes": selected_attributes,
        "base_values": values,
        "base_value_units": units,
        "answer": answer,
        "base_attribute_debug": selection_result,
    }


def _load_snapshot_cache_with_services(
    services: Dict[str, Any],
    parsed_query: Dict[str, Any] | None,
    snapshot_inventory: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cim_root = services["cim_root"]
    parsed_query = parsed_query or {}
    snapshot_inventory = snapshot_inventory or {}

    required_state_types = _extract_required_state_types(parsed_query)
    cim_snapshots: Dict[str, Any] = {}

    if _should_load_states(parsed_query):
        cim_snapshots = load_snapshots_for_time_window(
            root_folder=cim_root,
            start_time=parsed_query.get("time_start"),
            end_time=parsed_query.get("time_end"),
            snapshot_inventory=snapshot_inventory,
        )

        if not cim_snapshots:
            cim_snapshots = load_cim_snapshots(cim_root)

    if cim_snapshots and required_state_types:
        snapshot_cache = preprocess_snapshots_for_states(
            cim_snapshots=cim_snapshots,
            state_types=required_state_types,
        )
    else:
        snapshot_cache = {}

    snapshot_cache_summary = summarize_snapshot_cache(snapshot_cache)

    return {
        "status": "ok",
        "tool": "load_snapshot_cache",
        "cim_root": cim_root,
        "cim_snapshots": cim_snapshots,
        "required_state_types": required_state_types,
        "snapshot_cache": snapshot_cache,
        "snapshot_cache_summary": snapshot_cache_summary,
    }



def _query_cim_with_services(
    services: Dict[str, Any],
    user_input: str,
    snapshot_cache: Dict[str, Any] | None,
    network_index: Dict[str, Any] | None,
    parsed_query: Dict[str, Any] | None,
    classification: Dict[str, Any] | None,
) -> Dict[str, Any]:
    answer = handle_user_query(
        user_input=user_input,
        snapshot_cache=snapshot_cache or {},
        network_index=network_index or {},
        parsed_query=parsed_query,
        analysis_plan=classification,
    )

    return {
        "status": "ok",
        "tool": "query_cim",
        "cim_root": services["cim_root"],
        "answer": answer,
    }



def _summarize_cim_result_with_services(
    services: Dict[str, Any],
    result_payload: Dict[str, Any],
    user_input: str,
) -> Dict[str, Any]:
    snapshot_inventory = result_payload.get("snapshot_inventory") or {}
    network_index = result_payload.get("network_index") or {}
    cim_snapshots = result_payload.get("cim_snapshots") or {}
    snapshot_cache_summary = result_payload.get("snapshot_cache_summary") or {}

    debug = {
        "num_inventory_snapshots": len(snapshot_inventory.get("snapshots", []))
        if snapshot_inventory else 0,
        "index_source_snapshot": network_index.get("index_source_snapshot"),
        "index_source_time_str": network_index.get("index_source_time_str"),
        "required_state_types": result_payload.get("required_state_types", []) or [],
        "num_loaded_snapshots": len(cim_snapshots),
        "loaded_snapshot_names": list(cim_snapshots.keys()),
        "snapshot_cache_summary": snapshot_cache_summary,
        "parsed_query": result_payload.get("parsed_query", {}) or {},
        "resolved_object": result_payload.get("resolved_object"),
        "resolution_mode": result_payload.get("resolution_mode"),
        "equipment_resolution_debug": result_payload.get("equipment_resolution_debug", {}),
        "equipment_catalog_summary": result_payload.get("equipment_catalog_summary", {}),
    }

    return {
        "status": "ok",
        "tool": "summarize_cim_result",
        "cim_root": services["cim_root"],
        "user_input": user_input,
        "answer": result_payload.get("answer", ""),
        "debug": debug,
    }


# ------------------------------------------------------------------
# PUBLIC MCP-STYLE TOOLS (atomic)
# ------------------------------------------------------------------
def scan_snapshot_inventory(cim_root: Optional[str] = None) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services
    return _scan_snapshot_inventory_with_services(services)



def resolve_cim_object(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    return _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )



def list_equipment_of_type(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    return _list_equipment_of_type_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )




def read_cim_base_values(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    return _read_cim_base_values_with_services(
        services=services,
        user_input=user_input,
        resolved_object=resolve_result.get("resolved_object"),
    )


def load_snapshot_cache(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    return _load_snapshot_cache_with_services(
        services=services,
        parsed_query=resolve_result.get("parsed_query"),
        snapshot_inventory=scan_result.get("snapshot_inventory"),
    )



def query_cim(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    cache_result = _load_snapshot_cache_with_services(
        services=services,
        parsed_query=resolve_result.get("parsed_query"),
        snapshot_inventory=scan_result.get("snapshot_inventory"),
    )
    if cache_result.get("status") != "ok":
        return cache_result

    return _query_cim_with_services(
        services=services,
        user_input=user_input,
        snapshot_cache=cache_result.get("snapshot_cache"),
        network_index=scan_result.get("network_index"),
        parsed_query=resolve_result.get("parsed_query"),
        classification=None,
    )



def summarize_cim_result(
    result_payload: Dict[str, Any],
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    return _summarize_cim_result_with_services(
        services=services,
        result_payload=result_payload or {},
        user_input=user_input,
    )


# ------------------------------------------------------------------
# DOMAIN AGENT / REGISTRY HELPERS
# ------------------------------------------------------------------
def run_cim_agent(user_input: str, cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the CIM domain agent on a natural-language request.
    Local import avoids a circular dependency with the registry.
    """
    from cimpy.cimpy_time_analysis.cim_domain_agent import CIMDomainAgent

    agent = CIMDomainAgent(cim_root=_normalize_cim_root(cim_root))
    return agent.run(user_input)



def list_cim_tools(cim_root: Optional[str] = None) -> Dict[str, Any]:
    """
    List the currently available CIM domain tools and their metadata.
    Local import avoids a circular dependency with the registry.
    """
    from cimpy.cimpy_time_analysis.cim_tool_registry import CIMToolRegistry

    registry = CIMToolRegistry(cim_root=_normalize_cim_root(cim_root))
    return {
        "status": "ok",
        "tool": "list_cim_tools",
        "cim_root": _normalize_cim_root(cim_root),
        "available_tools": registry.list_tool_specs(),
    }


# ------------------------------------------------------------------
# BACKWARD-COMPATIBLE ALIASES
# ------------------------------------------------------------------
def scan_cim_snapshot_inventory(cim_root: Optional[str] = None) -> Dict[str, Any]:
    return scan_snapshot_inventory(cim_root=cim_root)



def load_cim_snapshot_cache(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    return load_snapshot_cache(user_input=user_input, cim_root=cim_root)



def execute_cim_query(
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    services = build_cim_services(cim_root=cim_root)
    if services.get("status") != "ok":
        return services

    scan_result = _scan_snapshot_inventory_with_services(services)
    if scan_result.get("status") != "ok":
        return scan_result

    resolve_result = _resolve_cim_object_with_services(
        services=services,
        user_input=user_input,
        network_index=scan_result.get("network_index"),
    )
    if resolve_result.get("status") != "ok":
        return resolve_result

    cache_result = _load_snapshot_cache_with_services(
        services=services,
        parsed_query=resolve_result.get("parsed_query"),
        snapshot_inventory=scan_result.get("snapshot_inventory"),
    )
    if cache_result.get("status") != "ok":
        return cache_result

    query_result = _query_cim_with_services(
        services=services,
        user_input=user_input,
        snapshot_cache=cache_result.get("snapshot_cache"),
        network_index=scan_result.get("network_index"),
        parsed_query=resolve_result.get("parsed_query"),
        classification=None,
    )
    if query_result.get("status") != "ok":
        return query_result

    summary_result = _summarize_cim_result_with_services(
        services=services,
        result_payload={
            **scan_result,
            **resolve_result,
            **cache_result,
            **query_result,
        },
        user_input=user_input,
    )
    if summary_result.get("status") != "ok":
        return summary_result

    return {
        "status": "ok",
        "tool": "execute_cim_query",
        "cim_root": _normalize_cim_root(cim_root),
        "answer": summary_result.get("answer", ""),
        "debug": {
            "scan_snapshot_inventory": scan_result,
            "resolve_cim_object": resolve_result,
            "load_snapshot_cache": cache_result,
            "query_cim": query_result,
            "summarize_cim_result": summary_result,
        },
    }



def summarize_cim_execution(
    result_payload: Dict[str, Any],
    user_input: str,
    cim_root: Optional[str] = None,
) -> Dict[str, Any]:
    return summarize_cim_result(
        result_payload=result_payload,
        user_input=user_input,
        cim_root=cim_root,
    )
