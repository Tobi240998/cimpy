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
from cimpy.cimpy_time_analysis.cim_queries import (
    query_equipment_metric_over_time,
    query_equipment_voltage_over_time,
    summarize_metric,
    summarize_voltage,
)


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


class ComparisonResolutionDecision(BaseModel):
    comparison_type: Optional[str] = Field(default=None)
    should_execute: bool = Field(default=False)
    rationale: str = Field(default="")


def _normalize_cim_root(cim_root: Optional[str] = None) -> str:
    return cim_root or CIM_ROOT


_COMPARISON_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "transformer_loading": {
        "equipment_types": ["PowerTransformer"],
        "required_state_type": "SvPowerFlow",
        "sv_metric": "S",
        "base_attributes": ["ratedS"],
        "comparison_mode": "upper_limit",
        "observed_value_mode": "abs_peak",
        "limit_attribute": "ratedS",
    },
    "generator_active_power_limit": {
        "equipment_types": ["SynchronousMachine", "AsynchronousMachine"],
        "required_state_type": "SvPowerFlow",
        "sv_metric": "P",
        "base_attributes": ["minOperatingP", "maxOperatingP"],
        "comparison_mode": "range_limit",
        "observed_value_mode": "abs_peak",
        "lower_attribute": "minOperatingP",
        "upper_attribute": "maxOperatingP",
    },
    "generator_reactive_power_limit": {
        "equipment_types": ["SynchronousMachine", "AsynchronousMachine"],
        "required_state_type": "SvPowerFlow",
        "sv_metric": "Q",
        "base_attributes": ["minQ", "maxQ"],
        "comparison_mode": "range_limit",
        "observed_value_mode": "peak",
        "lower_attribute": "minQ",
        "upper_attribute": "maxQ",
    },
    "voltage_vs_rated_voltage": {
        "equipment_types": [
            "PowerTransformer",
            "SynchronousMachine",
            "AsynchronousMachine",
            "ConformLoad",
            "EnergyConsumer",
            "BusbarSection",
            "ACLineSegment",
            "EquivalentInjection",
            "ExternalNetworkInjection",
        ],
        "required_state_type": "SvVoltage",
        "sv_metric": "U",
        "base_attributes": ["ratedU"],
        "comparison_mode": "upper_limit",
        "observed_value_mode": "peak",
        "limit_attribute": "ratedU",
    },
}


def _extract_scalar_base_value(value: Any, *, prefer_max: bool = True) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        candidates: List[float] = []
        for item in value:
            if isinstance(item, dict):
                raw = item.get("value")
            else:
                raw = item
            try:
                if raw is not None:
                    candidates.append(float(raw))
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates) if prefer_max else min(candidates)
    return None


def _extract_observed_series(query_result_data: Dict[str, Any] | None) -> List[float]:
    query_result_data = query_result_data or {}
    rows = query_result_data.get("rows", []) or []
    metric = str(query_result_data.get("metric") or "").upper()
    values: List[float] = []
    key = None
    if metric == "P":
        key = "active_power_MW"
    elif metric == "Q":
        key = "reactive_power_MVAr"
    elif metric == "S":
        key = "apparent_power_MVA"
    elif metric == "U":
        key = "voltage_kV"
    if key is None:
        return values
    for row in rows:
        try:
            raw = row.get(key) if isinstance(row, dict) else None
            if raw is not None:
                values.append(float(raw))
        except Exception:
            continue
    return values


def _looks_like_comparison_request(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False
    markers = [
        "überlast", "overload", "auslastung", "loading",
        "grenze", "grenzwert", "limit", "überschreit",
        "im verhältnis", "verglich", "compare", "against",
        "zulässig", "zulässige",
    ]
    return any(marker in text for marker in markers)


def _build_comparison_resolution_chain():
    parser = PydanticOutputParser(pydantic_object=ComparisonResolutionDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You classify a CIM comparison request into one allowed comparison type.\n"
            "Return only structured output.\n"
            "Choose only from the provided allowed comparison types.\n"
            "Do not invent attributes or comparison types.\n"
            "If the request is not clearly a comparison/limit check, set comparison_type=null and should_execute=false.\n\n"
            "Allowed comparison types:\n{comparison_definitions}\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Resolved equipment class: {equipment_class}\n"
            "Resolved equipment name: {equipment_name}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser


def _resolve_comparison_definition(user_input: str, resolved_object: Any, parsed_query: Dict[str, Any] | None = None) -> Dict[str, Any]:
    parsed_query = parsed_query or {}
    equipment_class = resolved_object.__class__.__name__ if resolved_object is not None else None
    equipment_name = _safe_name(resolved_object) or getattr(resolved_object, "mRID", None) if resolved_object is not None else None

    allowed = {
        name: spec for name, spec in _COMPARISON_DEFINITIONS.items()
        if equipment_class in spec.get("equipment_types", [])
    }
    if not allowed:
        return {
            "status": "error",
            "error": "no_allowed_comparisons_for_equipment",
            "details": f"No comparison definitions configured for equipment class '{equipment_class}'.",
        }

    try:
        chain, parser = _build_comparison_resolution_chain()
        definitions_text = "\n".join(
            f"- {name}: state_type={spec.get('required_state_type')} | sv_metric={spec.get('sv_metric')} | base_attributes={spec.get('base_attributes')} | mode={spec.get('comparison_mode')}"
            for name, spec in allowed.items()
        )
        decision = chain.invoke({
            "user_input": user_input,
            "equipment_class": equipment_class,
            "equipment_name": equipment_name,
            "comparison_definitions": definitions_text,
            "format_instructions": parser.get_format_instructions(),
        })
        comparison_type = decision.comparison_type
        if not decision.should_execute or comparison_type not in allowed:
            raise ValueError("comparison_not_safely_resolved")
        selected = dict(allowed[comparison_type])
        selected.update({
            "comparison_type": comparison_type,
            "status": "ok",
            "resolution_mode": "semantic_llm_match",
            "llm_decision": decision.model_dump() if hasattr(decision, "model_dump") else decision.dict(),
        })
        return selected
    except Exception as exc:
        text = (user_input or "").lower()
        fallback_type = None
        if equipment_class == "PowerTransformer" and any(m in text for m in ["überlast", "overload", "auslastung", "loading"]):
            fallback_type = "transformer_loading"
        elif equipment_class in {"SynchronousMachine", "AsynchronousMachine"} and any(m in text for m in ["überlast", "overload", "auslastung", "loading"]):
            fallback_type = "generator_active_power_limit"
        elif any(m in text for m in ["spannung", "voltage", "spannungsgrenze", "spannungsgrenzen"]):
            fallback_type = "voltage_vs_rated_voltage"
        if fallback_type and fallback_type in allowed:
            selected = dict(allowed[fallback_type])
            selected.update({
                "comparison_type": fallback_type,
                "status": "ok",
                "resolution_mode": "heuristic_fallback",
                "llm_error": str(exc),
            })
            return selected
        return {
            "status": "error",
            "error": "comparison_resolution_failed",
            "details": str(exc),
            "allowed_comparison_types": sorted(allowed.keys()),
        }


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



def _read_base_attribute_value(equipment_obj: Any, attr_name: str) -> Any:
    if equipment_obj is None or not attr_name:
        return None

    if hasattr(equipment_obj, attr_name):
        value = getattr(equipment_obj, attr_name, None)
        if value is not None:
            return value

    class_name = equipment_obj.__class__.__name__

    if class_name == "PowerTransformer":
        ends = getattr(equipment_obj, "PowerTransformerEnd", None) or []
        for end in ends:
            if hasattr(end, attr_name):
                value = getattr(end, attr_name, None)
                if value is not None:
                    return value

    if class_name == "SynchronousMachine":
        generating_unit = getattr(equipment_obj, "GeneratingUnit", None)
        if generating_unit is not None and hasattr(generating_unit, attr_name):
            value = getattr(generating_unit, attr_name, None)
            if value is not None:
                return value

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





def _read_cim_base_values_with_services(
    services: Dict[str, Any],
    user_input: str,
    resolved_object: Any,
    parsed_query: Dict[str, Any] | None = None,
    analysis_plan: Dict[str, Any] | None = None,
    requested_attributes: List[str] | None = None,
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

    if requested_attributes:
        available_attributes = [
            attr_name for attr_name in requested_attributes
            if _read_base_attribute_value(resolved_object, attr_name) is not None
        ]
        selection_result = {
            "selected_attributes": available_attributes,
            "resolution_mode": "preselected_attributes",
            "available_attributes": available_attributes,
            "requested_attributes": list(requested_attributes),
        }
    else:
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
    for attr_name in selected_attributes:
        values[attr_name] = _read_base_attribute_value(resolved_object, attr_name)

    print(f"selected_attributes: {selected_attributes}")
    print(f"base_values: {values}")

    equipment_name = _safe_name(resolved_object) or getattr(resolved_object, "mRID", None) or "<unbekannt>"
    parts = [f"{attr}={values.get(attr)!r}" for attr in selected_attributes]
    answer = f"Basiswerte für {equipment_name}: " + ", ".join(parts)

    return {
        "status": "ok",
        "tool": "read_cim_base_values",
        "cim_root": services["cim_root"],
        "selected_attributes": selected_attributes,
        "base_values": values,
        "answer": answer,
        "base_attribute_debug": selection_result,
    }


def _resolve_cim_comparison_with_services(
    services: Dict[str, Any],
    user_input: str,
    resolved_object: Any,
    parsed_query: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parsed_query = parsed_query or {}

    if resolved_object is None:
        return {
            "status": "error",
            "tool": "resolve_cim_comparison",
            "cim_root": services["cim_root"],
            "error": "missing_resolved_object",
            "answer": "Es wurde kein aufgelöstes CIM-Equipment für den Vergleich gefunden.",
        }

    resolution = _resolve_comparison_definition(
        user_input=user_input,
        resolved_object=resolved_object,
        parsed_query=parsed_query,
    )
    if resolution.get("status") != "ok":
        return {
            "status": "error",
            "tool": "resolve_cim_comparison",
            "cim_root": services["cim_root"],
            "error": resolution.get("error", "comparison_resolution_failed"),
            "answer": "Die gewünschte Vergleichslogik konnte nicht sicher aufgelöst werden.",
            "comparison_resolution_debug": resolution,
        }

    updated_parsed_query = dict(parsed_query)
    required_state_type = resolution.get("required_state_type")
    if required_state_type:
        updated_parsed_query["state_detected"] = [required_state_type]
    sv_metric = resolution.get("sv_metric")
    if sv_metric in {"P", "Q", "S"}:
        updated_parsed_query["metric"] = sv_metric

    return {
        "status": "ok",
        "tool": "resolve_cim_comparison",
        "cim_root": services["cim_root"],
        "comparison_resolution": resolution,
        "parsed_query": updated_parsed_query,
        "requested_base_attributes": list(resolution.get("base_attributes", []) or []),
        "comparison_resolution_debug": resolution,
        "answer": f"Vergleichslogik aufgelöst: {resolution.get('comparison_type')}",
    }


def _compare_cim_values_with_services(
    services: Dict[str, Any],
    resolved_object: Any,
    comparison_resolution: Dict[str, Any] | None,
    query_result_data: Dict[str, Any] | None,
    base_values: Dict[str, Any] | None,
) -> Dict[str, Any]:
    comparison_resolution = comparison_resolution or {}
    query_result_data = query_result_data or {}
    base_values = base_values or {}

    if comparison_resolution.get("status") != "ok":
        return {
            "status": "error",
            "tool": "compare_cim_values",
            "cim_root": services["cim_root"],
            "error": "missing_comparison_resolution",
            "answer": "Es liegt keine aufgelöste Vergleichsdefinition vor.",
        }

    observed_values = _extract_observed_series(query_result_data)
    if not observed_values:
        return {
            "status": "error",
            "tool": "compare_cim_values",
            "cim_root": services["cim_root"],
            "error": "missing_sv_values",
            "answer": "Es konnten keine SV-Werte für den Vergleich ermittelt werden.",
        }

    observed_mode = comparison_resolution.get("observed_value_mode") or "peak"
    if observed_mode == "abs_peak":
        observed_value = max(abs(v) for v in observed_values)
    else:
        observed_value = max(observed_values)

    comparison_type = comparison_resolution.get("comparison_type")
    result_payload: Dict[str, Any] = {
        "comparison_type": comparison_type,
        "observed_value": observed_value,
        "observed_value_mode": observed_mode,
        "sv_metric": query_result_data.get("metric"),
    }

    equipment_name = _safe_name(resolved_object) or getattr(resolved_object, "mRID", None) or "<unbekannt>"
    answer = None

    if comparison_resolution.get("comparison_mode") == "upper_limit":
        limit_attr = comparison_resolution.get("limit_attribute")
        limit_value = _extract_scalar_base_value(base_values.get(limit_attr), prefer_max=True)
        if limit_value is None:
            return {
                "status": "error",
                "tool": "compare_cim_values",
                "cim_root": services["cim_root"],
                "error": "missing_limit_value",
                "answer": f"Für den Vergleich fehlt der Basiswert '{limit_attr}'.",
            }
        ratio = (observed_value / limit_value * 100.0) if limit_value not in {None, 0} else None
        exceeds = bool(limit_value is not None and observed_value > limit_value)
        result_payload.update({
            "limit_attribute": limit_attr,
            "limit_value": limit_value,
            "ratio_percent": ratio,
            "exceeds_limit": exceeds,
        })
        status_text = "überschreitet" if exceeds else "liegt innerhalb von"
        answer = (
            f"Vergleich für {equipment_name}: beobachteter Wert={observed_value:.3f}, "
            f"{limit_attr}={limit_value:.3f} -> {status_text} dem Grenzwert"
        )
    elif comparison_resolution.get("comparison_mode") == "range_limit":
        lower_attr = comparison_resolution.get("lower_attribute")
        upper_attr = comparison_resolution.get("upper_attribute")
        lower_value = _extract_scalar_base_value(base_values.get(lower_attr), prefer_max=False)
        upper_value = _extract_scalar_base_value(base_values.get(upper_attr), prefer_max=True)
        if lower_value is None and upper_value is None:
            return {
                "status": "error",
                "tool": "compare_cim_values",
                "cim_root": services["cim_root"],
                "error": "missing_limit_value",
                "answer": f"Für den Vergleich fehlen die Basiswerte '{lower_attr}' und '{upper_attr}'.",
            }
        below = bool(lower_value is not None and observed_value < lower_value)
        above = bool(upper_value is not None and observed_value > upper_value)
        within = not below and not above
        result_payload.update({
            "lower_attribute": lower_attr,
            "lower_value": lower_value,
            "upper_attribute": upper_attr,
            "upper_value": upper_value,
            "within_limits": within,
            "below_lower_limit": below,
            "above_upper_limit": above,
        })
        if within:
            verdict = "liegt innerhalb der zulässigen Grenzen"
        elif above:
            verdict = "überschreitet die obere Grenze"
        else:
            verdict = "unterschreitet die untere Grenze"
        answer = (
            f"Vergleich für {equipment_name}: beobachteter Wert={observed_value:.3f}, "
            f"{lower_attr}={lower_value!r}, {upper_attr}={upper_value!r} -> {verdict}"
        )
    else:
        return {
            "status": "error",
            "tool": "compare_cim_values",
            "cim_root": services["cim_root"],
            "error": "unsupported_comparison_mode",
            "answer": "Der Vergleichsmodus wird aktuell nicht unterstützt.",
        }

    return {
        "status": "ok",
        "tool": "compare_cim_values",
        "cim_root": services["cim_root"],
        "comparison_result": result_payload,
        "answer": answer,
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
    resolved_object: Any | None = None,
    comparison_resolution: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parsed_query = dict(parsed_query or {})
    comparison_resolution = comparison_resolution or {}

    if comparison_resolution.get("status") == "ok":
        required_state_type = comparison_resolution.get("required_state_type")
        if required_state_type:
            parsed_query["state_detected"] = [required_state_type]
        sv_metric = comparison_resolution.get("sv_metric")
        if sv_metric in {"P", "Q", "S"}:
            parsed_query["metric"] = sv_metric

    answer = handle_user_query(
        user_input=user_input,
        snapshot_cache=snapshot_cache or {},
        network_index=network_index or {},
        parsed_query=parsed_query,
        analysis_plan=classification,
    )

    query_result_data: Dict[str, Any] = {}
    try:
        state_detected = parsed_query.get("state_detected", []) or []
        metric = parsed_query.get("metric")
        if resolved_object is not None and "SvVoltage" in state_detected:
            rows = query_equipment_voltage_over_time(
                snapshot_cache=snapshot_cache or {},
                network_index=network_index or {},
                equipment_obj=resolved_object,
            )
            query_result_data = {
                "query_mode": "voltage_over_time",
                "metric": "U",
                "rows": rows,
                "summary": summarize_voltage(rows),
            }
        elif resolved_object is not None and "SvPowerFlow" in state_detected and metric in {"P", "Q", "S"}:
            rows = query_equipment_metric_over_time(
                snapshot_cache=snapshot_cache or {},
                network_index=network_index or {},
                equipment_obj=resolved_object,
                metric=metric,
            )
            query_result_data = {
                "query_mode": "metric_over_time",
                "metric": metric,
                "rows": rows,
                "summary": summarize_metric(rows),
            }
    except Exception as exc:
        query_result_data = {
            "query_mode": "structured_query_failed",
            "error": str(exc),
        }

    return {
        "status": "ok",
        "tool": "query_cim",
        "cim_root": services["cim_root"],
        "parsed_query": parsed_query,
        "answer": answer,
        "query_result_data": query_result_data,
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
        "comparison_resolution": result_payload.get("comparison_resolution", {}),
        "comparison_result": result_payload.get("comparison_result", {}),
        "requested_base_attributes": result_payload.get("requested_base_attributes", []),
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
