from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cimpy.single_agent.llm_routing.config import CIM_ROOT
from cimpy.single_agent.cim.load_cim_data import (
    scan_snapshot_inventory as _scan_snapshot_inventory_raw,
    load_base_snapshot,
    build_network_index_from_snapshot,
    load_snapshots_for_time_window,
    load_cim_snapshots,
)
from cimpy.single_agent.cim.cim_snapshot_cache import (
    preprocess_snapshots_for_states,
    summarize_snapshot_cache,
)
from cimpy.single_agent.cim.llm_object_mapping import (
    interpret_user_query,
    interpret_equipment_type_query,
    resolve_requested_base_attributes,
    BASE_ATTRIBUTE_SPECS,
)
from cimpy.single_agent.cim.llm_cim_orchestrator import handle_user_query
from cimpy.single_agent.cim.langchain_llm import get_llm
from cimpy.single_agent.cim.cim_queries import (
    query_equipment_metric_over_time,
    query_equipment_voltage_over_time,
    summarize_metric,
    summarize_voltage,
)

from cimpy.single_agent.cim.cim_object_utils import collect_all_cim_objects

from cimpy.single_agent.cim.schemas import (
    EquipmentTypeDecision,
    EquipmentInstanceDecision,
    ParsedQueryNormalizationDecision,
    ComparisonResolutionDecision,
    FinalAnswerDecision,
    BaseAttributeIntentDecision,
    VoltageLimitSelectionDecision,
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


# setzt cim_route auf Default CIM_ROOT, falls nichts übergeben wurde 
def _normalize_cim_root(cim_root: Optional[str] = None) -> str:
    return cim_root or CIM_ROOT


_COMPARISON_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "bus_voltage_limits": {
        "equipment_types": ["TopologicalNode", "BusbarSection"],
        "required_state_type": "SvVoltage",
        "sv_metric": "voltage",
        "base_attributes": ["lowVoltageLimit", "highVoltageLimit"],
        "comparison_mode": "range_limit_per_timestamp",
        "observed_value_mode": "mean",
        "lower_attribute": "lowVoltageLimit",
        "upper_attribute": "highVoltageLimit",
    },

    "transformer_loading": {
        "equipment_types": ["PowerTransformer"],
        "required_state_type": "SvPowerFlow",
        "sv_metric": "S",
        "base_attributes": ["ratedS"],
        "comparison_mode": "upper_limit",
        "observed_value_mode": "abs_peak_timestamp",
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
_BASE_ATTRIBUTE_UNITS: Dict[str, Optional[str]] = {
    # Voltage
    "ratedU": "kV",
    "lowVoltageLimit": "kV",
    "highVoltageLimit": "kV",

    # Apparent Power
    "ratedS": "MVA",

    # Active Power
    "p": "MW",
    "initialP": "MW",
    "nominalP": "MW",
    "maxOperatingP": "MW",
    "minOperatingP": "MW",

    # Reactive Power
    "q": "MVAr",
    "maxQ": "MVAr",
    "minQ": "MVAr",

    # Impedance
    "r": "ohm",
    "r0": "ohm",
    "x": "ohm",
    "x0": "ohm",

    # Admittance
    "g": "S",
    "g0": "S",
    "b": "S",
    "b0": "S",

    # Dimensionless / meta
    "name": None,
    "mRID": None,
    "description": None,
    "type": None,
    "operatingMode": None,
    "connectionKind": None,
    "grounded": None,
    "endNumber": None,
    "phaseAngleClock": None,
}

# ordnet Base-Attributen Einheiten zu
def _base_attribute_unit(attr_name: str, equipment_obj: Any = None) -> Optional[str]:
    if not attr_name:
        return None

    if attr_name in _BASE_ATTRIBUTE_UNITS:
        return _BASE_ATTRIBUTE_UNITS[attr_name]

    if attr_name == "value" and equipment_obj is not None:
        class_name = equipment_obj.__class__.__name__
        if class_name == "VoltageLimit":
            return "kV"
        if class_name == "CurrentLimit":
            return "A"
        if class_name == "ActivePowerLimit":
            return "MW"
        if class_name == "ApparentPowerLimit":
            return "MVA"

    spec = BASE_ATTRIBUTE_SPECS.get(attr_name, {}) if isinstance(BASE_ATTRIBUTE_SPECS, dict) else {}
    unit = spec.get("unit")
    if isinstance(unit, str) and unit.strip():
        return unit.strip()

    return None

# formatiert Base-Werte für die Antwortausgabe mit Einheit
def _format_base_value_for_answer(attr_name: str, value: Any, equipment_obj: Any = None) -> str:
    unit = _base_attribute_unit(attr_name, equipment_obj)

    if isinstance(value, list):
        formatted_items: List[str] = []
        for item in value:
            if isinstance(item, dict):
                raw_value = item.get("value")
                raw_unit = item.get("unit") or unit
                if raw_value is None:
                    formatted_items.append(repr(item))
                elif raw_unit:
                    formatted_items.append(f"{raw_value!r} {raw_unit}")
                else:
                    formatted_items.append(f"{raw_value!r}")
            else:
                if unit and isinstance(item, (int, float)):
                    formatted_items.append(f"{item!r} {unit}")
                else:
                    formatted_items.append(repr(item))
        return "[" + ", ".join(formatted_items) + "]"

    if isinstance(value, dict):
        raw_value = value.get("value")
        raw_unit = value.get("unit") or unit
        if raw_value is not None:
            return f"{raw_value!r} {raw_unit}".strip() if raw_unit else f"{raw_value!r}"
        return repr(value)

    if unit and isinstance(value, (int, float)):
        return f"{value!r} {unit}"

    return repr(value)

# ergänzt rohe Base-Werte um value & unit 
def _build_base_values_with_units(values: Dict[str, Any], equipment_obj: Any = None) -> Dict[str, Any]:
    enriched: Dict[str, Any] = {}

    for attr_name, value in (values or {}).items():
        unit = _base_attribute_unit(attr_name, equipment_obj)
        enriched[attr_name] = {
            "value": value,
            "unit": unit,
        }

    return enriched


# macht aus Base-Werten einen skalaren Vergleichswert, auch wenn Listen/Dicts geliefert werden
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

# zieht numerische Messreihe aus query_result_data["rows"]
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

# extrahiert timestamp+value-Paare aus Query-Result für zeitbezogene Vergleiche 
def _extract_observed_rows(query_result_data: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    query_result_data = query_result_data or {}
    rows = query_result_data.get("rows", []) or []
    metric = str(query_result_data.get("metric") or "").upper()
    values: List[Dict[str, Any]] = []
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
        if not isinstance(row, dict):
            continue
        raw = row.get(key)
        if raw is None:
            continue
        try:
            numeric_value = float(raw)
        except Exception:
            continue

        timestamp = (
            row.get("timestamp")
            or row.get("time")
            or row.get("datetime")
            or row.get("ts")
        )

        values.append({
            "timestamp": timestamp,
            "value": numeric_value,
        })

    return values

# baut die strukturierte LLM-Chain für Comparison-Type-Auswahl
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

# Refactoring: Heuristik durch Keyword-Fallback
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



# normiert mRID/UUID auf interne canonical_id
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


# robuster Name-Zugriff
def _safe_name(value):
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


# robuster Description-Zugriff über mehrere mögliche Attribut
def _safe_description(value):
    if value is None:
        return None
    for attr in ("description", "desc", "shortName"):
        text = getattr(value, attr, None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None

# LLM-Prompt für Identifikation der Spannungsgrenzen 
def _build_voltage_limit_selection_chain():
    parser = PydanticOutputParser(pydantic_object=VoltageLimitSelectionDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You match VoltageLimit candidates for a CIM bus-like object.\n"
            "\n"
            "Your task:\n"
            "- Identify which candidate represents the LOWER voltage limit.\n"
            "- Identify which candidate represents the UPPER voltage limit.\n"
            "\n"
            "Rules:\n"
            "- You may only select candidate IDs that appear exactly in the provided list.\n"
            "- Do NOT invent candidate IDs.\n"
            "- You may select at most one candidate for each role.\n"
            "- The lower limit must be strictly less than the upper limit.\n"
            "- If the mapping is not safe or ambiguous, return null values and should_execute=false.\n"
            "\n"
            "{format_instructions}"
        ),
        (
            "user",
            "Bus context:\n{bus_context}\n\n"
            "VoltageLimit candidates:\n{candidate_text}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

# Auflösen der Spannungsgrenzen für Busse 
def _resolve_voltage_limit_values_for_bus(bus_obj: Any) -> Dict[str, Any]:
    """
    Resolve lower/upper voltage limit values for a bus-like object.

    Strategy:
    1) deterministically collect candidate VoltageLimit objects from the bus and attached terminals
    2) deterministically extract numeric candidate values and textual metadata
    3) use an LLM to classify which candidate is the lower and which is the upper voltage limit
    4) validate the returned candidate IDs before exposing values
    """
    if bus_obj is None:
        return {
            "lowVoltageLimit": None,
            "highVoltageLimit": None,
        }

    limit_objects: List[Any] = []

    def _append_limit(limit_obj: Any) -> None:
        if limit_obj is None:
            return
        if getattr(limit_obj.__class__, "__name__", "") != "VoltageLimit":
            return
        if limit_obj not in limit_objects:
            limit_objects.append(limit_obj)

    # 1) Direct limit sets on the bus-like object
    for limit_set in getattr(bus_obj, "OperationalLimitSet", None) or []:
        for limit_obj in (getattr(limit_set, "OperationalLimitValue", None) or []):
            _append_limit(limit_obj)
        for limit_obj in (getattr(limit_set, "OperationalLimit", None) or []):
            _append_limit(limit_obj)

    # 2) Limits via attached terminals
    terminals: List[Any] = []
    for attr_name in ("Terminals", "Terminal"):
        attr_value = getattr(bus_obj, attr_name, None)
        if isinstance(attr_value, list):
            terminals.extend(attr_value)
        elif attr_value is not None:
            terminals.append(attr_value)

    for terminal in terminals:
        for limit_set in getattr(terminal, "OperationalLimitSet", None) or []:
            for limit_obj in (getattr(limit_set, "OperationalLimitValue", None) or []):
                _append_limit(limit_obj)
            for limit_obj in (getattr(limit_set, "OperationalLimit", None) or []):
                _append_limit(limit_obj)

    candidates: List[Dict[str, Any]] = []
    candidate_value_lookup: Dict[str, float] = {}

    for idx, limit_obj in enumerate(limit_objects, start=1):
        value = _read_base_attribute_value(limit_obj, "value")
        if value is None:
            continue

        try:
            numeric_value = float(value)
        except Exception:
            continue

        candidate_id = f"cand_{idx}"
        candidate_row = {
            "candidate_id": candidate_id,
            "value": numeric_value,
            "name": getattr(limit_obj, "name", None),
            "description": getattr(limit_obj, "description", None),
            "limit_set_name": getattr(getattr(limit_obj, "OperationalLimitSet", None), "name", None),
        }
        candidates.append(candidate_row)
        candidate_value_lookup[candidate_id] = numeric_value

    if not candidates:
        return {
            "lowVoltageLimit": None,
            "highVoltageLimit": None,
        }

    # One candidate is not enough to infer both bounds safely.
    if len(candidates) == 1:
        return {
            "lowVoltageLimit": None,
            "highVoltageLimit": None,
        }

    candidate_text = "\n".join(
        f"- candidate_id={row['candidate_id']} | value={row['value']} | "
        f"name={row.get('name')} | description={row.get('description')} | "
        f"limit_set_name={row.get('limit_set_name')}"
        for row in candidates
    )

    bus_context = (
        f"class={bus_obj.__class__.__name__}, "
        f"name={getattr(bus_obj, 'name', None)}, "
        f"mRID={getattr(bus_obj, 'mRID', None)}"
    )

    try:
        chain, parser = _build_voltage_limit_selection_chain()
        decision = chain.invoke({
            "bus_context": bus_context,
            "candidate_text": candidate_text,
            "format_instructions": parser.get_format_instructions(),
        })

        if not getattr(decision, "should_execute", False):
            return {
                "lowVoltageLimit": None,
                "highVoltageLimit": None,
            }

        low_id = getattr(decision, "low_candidate_id", None)
        high_id = getattr(decision, "high_candidate_id", None)

        if low_id is not None and low_id not in candidate_value_lookup:
            low_id = None
        if high_id is not None and high_id not in candidate_value_lookup:
            high_id = None

        low_value = candidate_value_lookup.get(low_id) if low_id else None
        high_value = candidate_value_lookup.get(high_id) if high_id else None

        # Guard against inconsistent LLM outputs.
        if low_id and high_id and low_id == high_id:
            return {
                "lowVoltageLimit": None,
                "highVoltageLimit": None,
            }

        if low_value is not None and high_value is not None and low_value > high_value:
            return {
                "lowVoltageLimit": None,
                "highVoltageLimit": None,
            }

        return {
            "lowVoltageLimit": low_value,
            "highVoltageLimit": high_value,
        }

    except Exception:
        return {
            "lowVoltageLimit": None,
            "highVoltageLimit": None,
        }


# zentrale technische Attributauflösung von Equipment/Base-Objekten, inkl. Spezialfälle PowerTransformer, SynchronousMachine, Voltage-Limits
def _read_base_attribute_value(equipment_obj: Any, attr_name: str) -> Any:
    if equipment_obj is None or not attr_name:
        return None

    if attr_name in {"lowVoltageLimit", "highVoltageLimit"}:
        resolved_limits = _resolve_voltage_limit_values_for_bus(equipment_obj)
        value = resolved_limits.get(attr_name)
        if value is not None:
            return value
        return _resolve_voltage_limits_from_node(equipment_obj).get(attr_name)

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


# filtert aus allen CIM-Objekten die "echten" Equipment-Kandidaten heraus 
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


# baut deterministisch Equipment-Katalog aus Basissnapshot 
def _build_equipment_catalog(container) -> Dict[str, Any]:
    all_objects = collect_all_cim_objects(container)

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


# merged Katalog in network_index
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
# LLM-Auswahl eines CIM-Equipment-Typs aus Kandidatenliste
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


# LLM-Auswahl des konkreten Equipments aus der Kandidatenliste 
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


# extrahiert relevanten Teil zur Auswahl des Equipments 
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


# normalisiert LLM-Hints auf lowercase / string
def _normalize_hint_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


# LLM-basierte Equipmentauswahl 
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


# LLM-basierte Auswahl des konkreten Equipments 
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


# ruft die Query-Normalization-LLM-Chain auf
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


# steuert Equipment-Auflösung -> erst Typ, dann konkretes Equipment 
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
    equipment_name_hint = _normalize_hint_text(query_hint_result.get("equipment_name_hint"))

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

    all_instance_candidates = list(equipment_by_type.get(selected_type, []) or [])
    filtered_instance_candidates = all_instance_candidates
    name_hint_filter_mode = "not_used"

    if equipment_name_hint:
        exact_or_substring_matches = []
        token_overlap_matches = []

        hint_tokens = [tok for tok in equipment_name_hint.split() if tok]

        for obj in all_instance_candidates:
            candidate_name = _safe_name(obj)
            candidate_desc = _safe_description(obj)

            candidate_name_norm = _normalize_hint_text(candidate_name)
            candidate_desc_norm = _normalize_hint_text(candidate_desc)

            haystacks = [s for s in [candidate_name_norm, candidate_desc_norm] if s]

            if not haystacks:
                continue

            if any(equipment_name_hint == h or equipment_name_hint in h for h in haystacks):
                exact_or_substring_matches.append(obj)
                continue

            if hint_tokens and any(all(tok in h for tok in hint_tokens) for h in haystacks):
                token_overlap_matches.append(obj)

        if exact_or_substring_matches:
            filtered_instance_candidates = exact_or_substring_matches
            name_hint_filter_mode = "exact_or_substring"
        elif token_overlap_matches:
            filtered_instance_candidates = token_overlap_matches
            name_hint_filter_mode = "token_overlap"

    instance_result = _select_equipment_instance_with_llm(
        user_input=user_input,
        selected_type=selected_type,
        equipment_candidates=filtered_instance_candidates,
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
        "resolved_object": resolved_object,
        "type_llm_decision": type_result.get("llm_decision"),
        "instance_llm_decision": instance_result.get("llm_decision"),
        "resolution_mode": "catalog_type_then_instance_llm",
        "query_hint_result": query_hint_result,
        "name_hint_filter_mode": name_hint_filter_mode,
        "num_instance_candidates_before_filter": len(all_instance_candidates),
        "num_instance_candidates_after_filter": len(filtered_instance_candidates),
    }


# ------------------------------------------------------------------
# INTERNAL HELPERS 
# ------------------------------------------------------------------
# leitet aus parsed_query["state_detected"] die tatsächlich zu ladenden State-Typen ab
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


# boolescher Shortcut, ob überhaupt State-Snapshots geladen werden sollen
def _should_load_states(parsed_query: Dict[str, Any] | None) -> bool:
    return len(_extract_required_state_types(parsed_query)) > 0


# ------------------------------------------------------------------
# LOW-LEVEL TOOL IMPLEMENTATIONS
# ------------------------------------------------------------------
# lädt/scannt Snapshot-Inventar, Base-Snapshot, Netzwerkindex, Equipment-Katalog
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


# Query parsen, Equipment-Typ / Equipment-Instanz auflösen
def _resolve_cim_object_with_services(
    services: Dict[str, Any],
    user_input: str,
    network_index: Dict[str, Any] | None,
    classification: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    classification = classification or {}
    request_mode = str(classification.get("request_mode") or "").strip()

    require_time_window = request_mode in {"standard_sv", "standard_comparison"}
    parsed_query = interpret_user_query(
        user_input=user_input,
        network_index=network_index,
        require_time_window=require_time_window,
        is_topology_query=(classification.get("intent") == "topology_query"),
    )

    if not isinstance(parsed_query, dict):
        parsed_query = {}

    parsed_query["intent"] = classification.get("intent")
    parsed_query["request_mode"] = classification.get("request_mode")
    parsed_query["is_topology_query"] = classification.get("intent") == "topology_query"
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

# ermittelt einen Equipment-Typ und listet alle passenden Objekte aus dem deterministischen Katalog
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


# Fallback für Bus Voltage-Limit Vergleich; Refactoring: Heuristik und ineffizient, dass nochmal alles geladen wird 
def _resolve_voltage_limit_values_via_global_lookup(
    services: Dict[str, Any],
    bus_obj: Any,
) -> Dict[str, Any]:
    """
    Fallback for bus voltage-limit comparisons:
    resolve the concrete VoltageLimit objects globally from the base snapshot and
    read their direct `.value` attribute, mirroring the successful standalone
    single-attribute path that resolves to VoltageLimit and reads `value`.
    """
    if bus_obj is None:
        return {
            "lowVoltageLimit": None,
            "highVoltageLimit": None,
        }

    try:
        cim_root = services["cim_root"]
        inventory = _scan_snapshot_inventory_raw(cim_root)
        base_snapshot = load_base_snapshot(
            root_folder=cim_root,
            snapshot_inventory=inventory,
        )
        all_objects = collect_all_cim_objects(base_snapshot)
    except Exception:
        return {
            "lowVoltageLimit": None,
            "highVoltageLimit": None,
        }

    bus_name = (_safe_name(bus_obj) or "").strip().lower()
    bus_mrid = (getattr(bus_obj, "mRID", None) or "").strip().lower()

    lows: List[float] = []
    highs: List[float] = []

    for obj in all_objects:
        if getattr(obj.__class__, "__name__", "") != "VoltageLimit":
            continue

        value = _read_base_attribute_value(obj, "value")
        if value is None:
            continue

        limit_set = getattr(obj, "OperationalLimitSet", None)
        terminal = getattr(limit_set, "Terminal", None) if limit_set is not None else None
        conducting_equipment = getattr(terminal, "ConductingEquipment", None) if terminal is not None else None
        topological_node = getattr(terminal, "TopologicalNode", None) if terminal is not None else None

        related_names = [
            getattr(obj, "name", None),
            getattr(obj, "description", None),
            getattr(limit_set, "name", None) if limit_set is not None else None,
            _safe_name(conducting_equipment) if conducting_equipment is not None else None,
            _safe_name(topological_node) if topological_node is not None else None,
            getattr(conducting_equipment, "mRID", None) if conducting_equipment is not None else None,
            getattr(topological_node, "mRID", None) if topological_node is not None else None,
        ]
        label = " ".join(str(x) for x in related_names if isinstance(x, str) and x.strip()).lower()

        bus_match = False
        if bus_name and bus_name in label:
            bus_match = True
        if not bus_match and bus_mrid and bus_mrid in label:
            bus_match = True
        if not bus_match:
            continue

        try:
            numeric_value = float(value)
        except Exception:
            continue

        if any(token in label for token in ["low limit", "lower", "minimum", "min ", "unter", "low voltage"]):
            lows.append(numeric_value)
            continue
        if any(token in label for token in ["high limit", "upper", "maximum", "max ", "ober", "high voltage"]):
            highs.append(numeric_value)
            continue

    return {
        "lowVoltageLimit": min(lows) if lows else None,
        "highVoltageLimit": max(highs) if highs else None,
    }

def _build_base_attribute_intent_chain():
    parser = PydanticOutputParser(pydantic_object=BaseAttributeIntentDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You classify which CIM base attributes are explicitly requested by the user.\n"
            "Return only structured output.\n"
            "You may only choose attribute names from the provided allowed attribute list.\n"
            "Do not invent attribute names.\n"
            "If the user clearly asks for voltage limits / Spannungsgrenzen / zulässige Spannungsgrenzen,\n"
            "prefer lowVoltageLimit and highVoltageLimit when they are available in the allowed list.\n"
            "Set should_use_preselected_attributes=true only if the requested attributes are clear enough\n"
            "to skip generic semantic attribute matching.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Equipment class: {equipment_class}\n"
            "Allowed attribute names:\n{allowed_attributes}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser


def _resolve_preselected_base_attributes_with_llm(
    user_input: str,
    resolved_object: Any,
) -> Dict[str, Any]:
    if resolved_object is None:
        return {
            "status": "error",
            "error": "missing_resolved_object",
            "requested_attributes": [],
            "should_use_preselected_attributes": False,
        }

    allowed_attributes = [
        "lowVoltageLimit",
        "highVoltageLimit",
    ]

    try:
        chain, parser = _build_base_attribute_intent_chain()
        decision = chain.invoke({
            "user_input": user_input,
            "equipment_class": resolved_object.__class__.__name__,
            "allowed_attributes": "\n".join(f"- {attr}" for attr in allowed_attributes),
            "format_instructions": parser.get_format_instructions(),
        })

        requested_attributes = [
            attr for attr in (decision.requested_attributes or [])
            if attr in allowed_attributes
        ]

        return {
            "status": "ok",
            "requested_attributes": requested_attributes,
            "should_use_preselected_attributes": bool(decision.should_use_preselected_attributes),
            "llm_decision": decision.model_dump() if hasattr(decision, "model_dump") else decision.dict(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": "base_attribute_intent_resolution_failed",
            "details": str(exc),
            "requested_attributes": [],
            "should_use_preselected_attributes": False,
        }

def _resolve_base_attribute_with_expanded_search(
    services: Dict[str, Any],
    resolved_object: Any,
    attr_name: str,
) -> Dict[str, Any]:
    """
    Resolve a selected base attribute in multiple stages:

    1) direct / canonical reader
    2) expanded structural lookup for known complex attribute families
    3) structured not_found result
    """
    searched_scopes: List[str] = []

    # 1) Canonical direct reader
    searched_scopes.append("direct_or_canonical_reader")
    try:
        value = _read_base_attribute_value(resolved_object, attr_name)
    except Exception as exc:
        return {
            "status": "error",
            "attribute": attr_name,
            "error": "base_attribute_read_failed",
            "details": str(exc),
            "searched_scopes": searched_scopes,
        }

    if value is not None:
        return {
            "status": "ok",
            "attribute": attr_name,
            "value": value,
            "searched_scopes": searched_scopes,
            "resolution_path": "direct_or_canonical_reader",
        }

    # 2) Expanded structural lookup for known complex attributes
    if attr_name in {"lowVoltageLimit", "highVoltageLimit"}:
        searched_scopes.append("expanded_voltage_limit_lookup")

        try:
            expanded = _resolve_voltage_limit_values_via_global_lookup(services, resolved_object)
            expanded_value = expanded.get(attr_name)
            if expanded_value is not None:
                return {
                    "status": "ok",
                    "attribute": attr_name,
                    "value": expanded_value,
                    "searched_scopes": searched_scopes,
                    "resolution_path": "expanded_voltage_limit_lookup",
                }
        except Exception as exc:
            return {
                "status": "error",
                "attribute": attr_name,
                "error": "expanded_voltage_limit_lookup_failed",
                "details": str(exc),
                "searched_scopes": searched_scopes,
            }

    # 3) No result
    return {
        "status": "not_found",
        "attribute": attr_name,
        "value": None,
        "searched_scopes": searched_scopes,
        "resolution_path": "not_found",
    }

def _is_simple_readable_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _extract_simple_values_from_object(obj: Any) -> Dict[str, Any]:
    """
    Extract simple readable scalar values from an object.
    Only returns primitive fields, not nested objects/lists.
    """
    result: Dict[str, Any] = {}

    if obj is None:
        return result

    # Prefer a few semantically useful names first if present
    preferred_names = [
        "value",
        "nominalVoltage",
        "ratedU",
        "ratedS",
        "name",
        "mRID",
        "description",
        "shortName",
        "energyIdentCodeEic",
        "ipMax",
        "aggregate",
    ]

    for attr_name in preferred_names:
        if not hasattr(obj, attr_name):
            continue
        try:
            attr_value = getattr(obj, attr_name, None)
        except Exception:
            continue
        if attr_value is None:
            continue
        if _is_simple_readable_value(attr_value):
            result[attr_name] = attr_value

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        for attr_name in obj_dict.keys():
            attr_name = str(attr_name)
            if not attr_name or attr_name.startswith("_"):
                continue
            if attr_name in result:
                continue
            try:
                attr_value = getattr(obj, attr_name, None)
            except Exception:
                continue
            if attr_value is None:
                continue
            if _is_simple_readable_value(attr_value):
                result[attr_name] = attr_value

    return result


def _iter_navigation_candidates_from_object(obj: Any) -> List[tuple[str, Any]]:
    """
    Return navigable child references from an object.
    Includes direct object refs and list refs, but not simple scalar values.
    """
    children: List[tuple[str, Any]] = []

    if obj is None:
        return children

    obj_dict = getattr(obj, "__dict__", None)
    if not isinstance(obj_dict, dict):
        return children

    for attr_name in obj_dict.keys():
        attr_name = str(attr_name)
        if not attr_name or attr_name.startswith("_"):
            continue

        try:
            attr_value = getattr(obj, attr_name, None)
        except Exception:
            continue

        if attr_value is None:
            continue
        if _is_simple_readable_value(attr_value):
            continue
        if callable(attr_value):
            continue

        if isinstance(attr_value, list):
            if attr_value:
                children.append((attr_name, attr_value))
            continue

        children.append((attr_name, attr_value))

    return children


def _resolve_candidate_value_deep(
    candidate_value: Any,
    target_attr: str | None = None,
) -> Dict[str, Any]:
    """
    Generic graph/value traversal for a candidate result.

    Strategy:
    - simple scalar -> return directly
    - object -> inspect simple readable fields, then recurse into child refs
    - list -> inspect items recursively
    """
    visited: set[int] = set()

    def _walk(value: Any, depth: int, path: List[str]) -> Dict[str, Any]:
        if value is None:
            return {
                "status": "not_found",
                "value": None,
                "resolution_path": " -> ".join(path) if path else "none",
            }

        if _is_simple_readable_value(value):
            return {
                "status": "ok",
                "value": value,
                "resolution_path": " -> ".join(path) if path else "direct_scalar",
            }

        if depth >= max_depth:
            return {
                "status": "not_found",
                "value": None,
                "resolution_path": " -> ".join(path + ["max_depth_reached"]),
            }

        obj_id = id(value)
        if obj_id in visited:
            return {
                "status": "not_found",
                "value": None,
                "resolution_path": " -> ".join(path + ["cycle_detected"]),
            }
        visited.add(obj_id)

        if isinstance(value, list):
            for idx, item in enumerate(value[:max_list_items]):
                result = _walk(item, depth + 1, path + [f"[{idx}]"])
                if result.get("status") == "ok":
                    return result
            return {
                "status": "not_found",
                "value": None,
                "resolution_path": " -> ".join(path + ["list_exhausted"]),
            }

        # Object: first try simple readable fields
        simple_fields = _extract_simple_values_from_object(value)
        preferred_field_order = [
            "value",
            "nominalVoltage",
            "ratedU",
            "ratedS",
            "name",
            "mRID",
            "description",
            "shortName",
            "energyIdentCodeEic",
            "ipMax",
            "aggregate",
        ]

        for field_name in preferred_field_order:
            if field_name in simple_fields:
                return {
                    "status": "ok",
                    "value": simple_fields[field_name],
                    "resolved_field": field_name,
                    "resolution_path": " -> ".join(path + [field_name]),
                }

        for field_name, field_value in simple_fields.items():
            return {
                "status": "ok",
                "value": field_value,
                "resolved_field": field_name,
                "resolution_path": " -> ".join(path + [field_name]),
            }

        # Then recurse into navigable references
        for child_name, child_value in _iter_navigation_candidates_from_object(value):
            result = _walk(child_value, depth + 1, path + [child_name])
            if result.get("status") == "ok":
                return result

        return {
            "status": "not_found",
            "value": None,
            "resolution_path": " -> ".join(path + ["no_readable_value_found"]),
        }

    return _walk(candidate_value, 0, [])


def _resolve_base_candidate_agentically(
    services: Dict[str, Any],
    resolved_object: Any,
    candidate_name: str,
) -> Dict[str, Any]:
    searched_scopes: List[str] = []

    # --------------------------------------------------------------
    # 1) Direct / canonical reader path
    # --------------------------------------------------------------
    searched_scopes.append("direct_or_canonical_reader")
    try:
        direct_value = _read_base_attribute_value(resolved_object, candidate_name)
        if direct_value is not None:
            deep_result = _resolve_candidate_value_deep(direct_value)

            if deep_result.get("status") == "ok":
                return {
                    "status": "ok",
                    "candidate": candidate_name,
                    "resolved_attribute": candidate_name,
                    "value": deep_result.get("value"),
                    "searched_scopes": searched_scopes,
                    "resolution_path": f"direct_or_canonical_reader -> {deep_result.get('resolution_path')}",
                    "resolved_field": deep_result.get("resolved_field"),
                }
    except Exception as exc:
        return {
            "status": "error",
            "candidate": candidate_name,
            "error": "direct_or_canonical_reader_failed",
            "details": str(exc),
            "searched_scopes": searched_scopes,
        }

    # --------------------------------------------------------------
    # 2) Direct raw object attribute path
    # --------------------------------------------------------------
    searched_scopes.append("direct_object_attribute")
    try:
        if hasattr(resolved_object, candidate_name):
            raw_value = getattr(resolved_object, candidate_name, None)
            if raw_value is not None:
                deep_result = _resolve_candidate_value_deep(raw_value)

                if deep_result.get("status") == "ok":
                    return {
                        "status": "ok",
                        "candidate": candidate_name,
                        "resolved_attribute": candidate_name,
                        "value": deep_result.get("value"),
                        "searched_scopes": searched_scopes,
                        "resolution_path": f"direct_object_attribute -> {deep_result.get('resolution_path')}",
                        "resolved_field": deep_result.get("resolved_field"),
                    }
    except Exception as exc:
        return {
            "status": "error",
            "candidate": candidate_name,
            "error": "direct_object_attribute_failed",
            "details": str(exc),
            "searched_scopes": searched_scopes,
        }

    # --------------------------------------------------------------
    # 3) Expanded structural lookup from neighboring graph/object refs
    # --------------------------------------------------------------
    searched_scopes.append("expanded_structural_lookup")
    try:
        for child_name, child_value in _iter_navigation_candidates_from_object(resolved_object):
            deep_result = _resolve_candidate_value_deep(child_value)

            if deep_result.get("status") != "ok":
                continue

            resolved_field = deep_result.get("resolved_field")

            # Do not accept arbitrary metadata from neighboring objects
            # as value for the requested base attribute.
            if resolved_field != candidate_name:
                continue

            return {
                "status": "ok",
                "candidate": candidate_name,
                "resolved_attribute": candidate_name,
                "value": deep_result.get("value"),
                "searched_scopes": searched_scopes,
                "resolution_path": f"expanded_structural_lookup -> {child_name} -> {deep_result.get('resolution_path')}",
                "resolved_field": resolved_field,
            }

    except Exception as exc:
        # Important: do not abort here. Let step 4 fallback run.
        searched_scopes.append(f"expanded_structural_lookup_error: {exc}")

    # --------------------------------------------------------------
    # 4) Optional known heavy fallback for voltage-limit family
    #    Keep this because it already exists and is useful, but only
    #    after the generic search stages above.
    # --------------------------------------------------------------
    if candidate_name in {"OperationalLimitSet", "lowVoltageLimit", "highVoltageLimit"}:
        searched_scopes.append("expanded_voltage_limit_lookup")
        try:
            expanded = _resolve_voltage_limit_values_via_global_lookup(services, resolved_object)
            low = expanded.get("lowVoltageLimit")
            high = expanded.get("highVoltageLimit")

            if candidate_name == "lowVoltageLimit" and low is not None:
                return {
                    "status": "ok",
                    "candidate": candidate_name,
                    "resolved_attribute": "lowVoltageLimit",
                    "value": low,
                    "searched_scopes": searched_scopes,
                    "resolution_path": "expanded_voltage_limit_lookup -> lowVoltageLimit",
                }

            if candidate_name == "highVoltageLimit" and high is not None:
                return {
                    "status": "ok",
                    "candidate": candidate_name,
                    "resolved_attribute": "highVoltageLimit",
                    "value": high,
                    "searched_scopes": searched_scopes,
                    "resolution_path": "expanded_voltage_limit_lookup -> highVoltageLimit",
                }

            if candidate_name == "OperationalLimitSet":
                result_values = {}
                if low is not None:
                    result_values["lowVoltageLimit"] = low
                if high is not None:
                    result_values["highVoltageLimit"] = high

                if result_values:
                    return {
                        "status": "ok",
                        "candidate": candidate_name,
                        "resolved_attribute": "voltage_limits_from_operational_limit_set",
                        "value": result_values,
                        "searched_scopes": searched_scopes,
                        "resolution_path": "expanded_voltage_limit_lookup -> voltage_limits_from_operational_limit_set",
                    }
        except Exception as exc:
            return {
                "status": "error",
                "candidate": candidate_name,
                "error": "expanded_voltage_limit_lookup_failed",
                "details": str(exc),
                "searched_scopes": searched_scopes,
            }

    # --------------------------------------------------------------
    # 5) Structured not_found
    # --------------------------------------------------------------
    return {
        "status": "not_found",
        "candidate": candidate_name,
        "value": None,
        "searched_scopes": searched_scopes,
        "resolution_path": "not_found",
    }

# liest statische Base-/Nameplate-Attribute eines aufgelösten Objekts
def _read_cim_base_values_with_services(
    services: Dict[str, Any],
    user_input: str,
    resolved_object: Any,
    parsed_query: Dict[str, Any] | None = None,
    analysis_plan: Dict[str, Any] | None = None,
    requested_attributes: List[str] | None = None,
) -> Dict[str, Any]:
    parsed_query = parsed_query or {}


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
                "selected_candidates": [],
            },
        }


    preselected_values: Dict[str, Any] | None = None
    selection_result: Dict[str, Any] = {}
    semantic_preselection = None

    # ------------------------------------------------------------------
    # 1) Preselected path:
    #    - explicit requested_attributes from caller
    #    - or LLM-based semantic preselection for known complex intents
    # ------------------------------------------------------------------
    effective_requested_attributes = list(requested_attributes or [])

    if not effective_requested_attributes:
        semantic_preselection = _resolve_preselected_base_attributes_with_llm(
            user_input=user_input,
            resolved_object=resolved_object,
        )

        if (
            semantic_preselection.get("status") == "ok"
            and semantic_preselection.get("should_use_preselected_attributes")
            and semantic_preselection.get("requested_attributes")
        ):
            effective_requested_attributes = list(
                semantic_preselection.get("requested_attributes") or []
            )

    if effective_requested_attributes:
        preselected_values = {}
        preselected_resolution_debug: Dict[str, Any] = {}

        for attr_name in effective_requested_attributes:
            resolution = _resolve_base_candidate_agentically(
                services=services,
                resolved_object=resolved_object,
                candidate_name=attr_name,
            )
            preselected_resolution_debug[attr_name] = resolution

            if resolution.get("status") == "ok":
                resolved_attr = resolution.get("resolved_attribute") or attr_name
                resolved_value = resolution.get("value")

                # If a structural candidate resolves to multiple final attributes,
                # merge them into preselected_values.
                if isinstance(resolved_value, dict):
                    for sub_attr, sub_value in resolved_value.items():
                        if sub_value is not None:
                            preselected_values[sub_attr] = sub_value
                else:
                    preselected_values[resolved_attr] = resolved_value

        available_attributes = [
            attr_name for attr_name, attr_value in preselected_values.items()
            if attr_value is not None
        ]

        selection_result = {
            "selected_attributes": available_attributes,
            "selected_candidates": list(effective_requested_attributes),
            "resolution_mode": "preselected_attributes",
            "available_attributes": available_attributes,
            "requested_attributes": list(effective_requested_attributes),
            "preselected_values": preselected_values,
            "preselected_resolution_debug": preselected_resolution_debug,
        }

        if semantic_preselection is not None:
            selection_result["semantic_preselection"] = semantic_preselection

    # ------------------------------------------------------------------
    # 2) Generic candidate matching path
    # ------------------------------------------------------------------
    else:
        selection_result = resolve_requested_base_attributes(
            user_input=user_input,
            equipment_obj=resolved_object,
        )

    print(f"base_attribute_debug: {selection_result}")

    selected_attributes = list(selection_result.get("selected_attributes", []) or [])
    selected_candidates = list(selection_result.get("selected_candidates", []) or [])

    # Legacy compatibility:
    # if the resolver still returns selected_attributes only, reuse them as candidates.
    if not selected_candidates and selected_attributes:
        selected_candidates = list(selected_attributes)

    if not selected_candidates and not selected_attributes:
        return {
            "status": "error",
            "tool": "read_cim_base_values",
            "cim_root": services["cim_root"],
            "error": "no_matching_base_attributes",
            "answer": "Für die Anfrage konnten keine passenden technischen Basisattribute gefunden werden.",
            "base_attribute_debug": selection_result,
        }

    # ------------------------------------------------------------------
    # 3) Agentic candidate resolution loop
    #    - try each candidate
    #    - direct / canonical read
    #    - expanded search
    #    - merge final resolved attributes
    # ------------------------------------------------------------------
    values: Dict[str, Any] = {}
    attribute_resolution_debug: Dict[str, Any] = {}

    # First keep already resolved preselected values if available.
    if preselected_values:
        for attr_name, attr_value in preselected_values.items():
            if attr_value is not None:
                values[attr_name] = attr_value
                attribute_resolution_debug[attr_name] = {
                    "status": "ok",
                    "attribute": attr_name,
                    "value": attr_value,
                    "searched_scopes": ["preselected_attributes"],
                    "resolution_path": "preselected_attributes",
                }

    # Then resolve all generic candidates.
    for candidate_name in selected_candidates:
        # avoid re-resolving already covered direct attributes
        if candidate_name in values:
            continue

        resolution = _resolve_base_candidate_agentically(
            services=services,
            resolved_object=resolved_object,
            candidate_name=candidate_name,
        )
        attribute_resolution_debug[candidate_name] = resolution

        if resolution.get("status") != "ok":
            continue

        resolved_attr = resolution.get("resolved_attribute") or candidate_name
        resolved_value = resolution.get("value")

        # Structural candidate can resolve to multiple final values.
        if isinstance(resolved_value, dict):
            for sub_attr, sub_value in resolved_value.items():
                if sub_value is not None:
                    values[sub_attr] = sub_value
        else:
            if resolved_value is not None:
                values[resolved_attr] = resolved_value

    # Final selected attributes = all actually resolved readable values
    selected_attributes = [
        attr_name for attr_name, attr_value in values.items()
        if attr_value is not None
    ]

    if not selected_attributes:
        return {
            "status": "error",
            "tool": "read_cim_base_values",
            "cim_root": services["cim_root"],
            "error": "no_resolved_base_values_found",
            "answer": "Für die angefragten Basisattribute konnten keine lesbaren Werte gefunden werden.",
            "base_attribute_debug": selection_result,
            "attribute_resolution_debug": attribute_resolution_debug,
        }

    enriched_base_values = _build_base_values_with_units(values, resolved_object)

    print(f"selected_candidates: {selected_candidates}")
    print(f"selected_attributes: {selected_attributes}")
    print(f"base_values: {values}")
    print(f"base_values_with_units: {enriched_base_values}")

    equipment_name = _safe_name(resolved_object) or getattr(resolved_object, "mRID", None) or "<unbekannt>"
    parts = [
        f"{attr}={_format_base_value_for_answer(attr, values.get(attr), resolved_object)}"
        for attr in selected_attributes
    ]
    answer = f"Basiswerte für {equipment_name}: " + ", ".join(parts)

    return {
        "status": "ok",
        "tool": "read_cim_base_values",
        "cim_root": services["cim_root"],
        "selected_candidates": selected_candidates,
        "selected_attributes": selected_attributes,
        "base_values": values,
        "base_values_with_units": enriched_base_values,
        "answer": answer,
        "base_attribute_debug": selection_result,
        "attribute_resolution_debug": attribute_resolution_debug,
    }

# mappt VoltageLimit-Objekt zurück auf Bus/TopologicalNode für Vergleichslogik
def _resolve_comparison_target_object(resolved_object: Any) -> Dict[str, Any]:
    if resolved_object is None:
        return {
            "status": "error",
            "error": "missing_resolved_object",
        }

    class_name = resolved_object.__class__.__name__

    if class_name != "VoltageLimit":
        return {
            "status": "ok",
            "resolved_object": resolved_object,
            "resolution_mode": "unchanged",
        }

    try:
        limit_set = getattr(resolved_object, "OperationalLimitSet", None)
        if limit_set is not None:
            terminal = getattr(limit_set, "Terminal", None)
            if terminal is not None:
                conducting_equipment = getattr(terminal, "ConductingEquipment", None)
                if conducting_equipment is not None and conducting_equipment.__class__.__name__ in {"BusbarSection", "TopologicalNode"}:
                    return {
                        "status": "ok",
                        "resolved_object": conducting_equipment,
                        "resolution_mode": "voltage_limit_to_terminal_conducting_equipment",
                        "source_limit_object": resolved_object,
                    }

                topological_node = getattr(terminal, "TopologicalNode", None)
                if topological_node is not None and topological_node.__class__.__name__ in {"TopologicalNode", "BusbarSection"}:
                    return {
                        "status": "ok",
                        "resolved_object": topological_node,
                        "resolution_mode": "voltage_limit_to_terminal_topological_node",
                        "source_limit_object": resolved_object,
                    }

        return {
            "status": "ok",
            "resolved_object": resolved_object,
            "resolution_mode": "voltage_limit_no_parent_found",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": "comparison_target_resolution_failed",
            "details": str(exc),
            "resolved_object": resolved_object,
        }

# bildet eine User-Anfrage auf einen unterstützten Vergleichstyp ab, z. B. Trafo-Auslastung oder Spannungsgrenzenvergleich; Refactoring: Heuristik bei Fallback 
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

    target_resolution = _resolve_comparison_target_object(resolved_object)
    comparison_target_object = target_resolution.get("resolved_object", resolved_object)

    resolution = _resolve_comparison_definition(
        user_input=user_input,
        resolved_object=comparison_target_object,
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
        "comparison_resolution_debug": {
            **resolution,
            "comparison_target_resolution": target_resolution,
        },
        "resolved_object": comparison_target_object,
        "answer": f"Vergleichslogik aufgelöst: {resolution.get('comparison_type')}",
    }

# vergleicht Query-Ergebnis und Base-Werte, berechnet Abweichung / Grenzverletzung / Range-Check
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

    comparison_type = comparison_resolution.get("comparison_type")
    comparison_mode = comparison_resolution.get("comparison_mode")
    observed_mode = comparison_resolution.get("observed_value_mode") or "peak"

    result_payload: Dict[str, Any] = {
        "comparison_type": comparison_type,
        "observed_value_mode": observed_mode,
        "sv_metric": query_result_data.get("metric"),
    }

    if comparison_mode == "range_limit_per_timestamp":
        observed_rows = _extract_observed_rows(query_result_data)
        if not observed_rows:
            return {
                "status": "error",
                "tool": "compare_cim_values",
                "cim_root": services["cim_root"],
                "error": "missing_sv_values",
                "answer": "Es konnten keine SV-Werte für den Vergleich ermittelt werden.",
            }
        observed_value = observed_rows[-1]["value"]
        result_payload["observed_value"] = observed_value
    else:
        if observed_mode in {"abs_peak_timestamp", "peak_timestamp"}:
            observed_rows = _extract_observed_rows(query_result_data)
            if not observed_rows:
                return {
                    "status": "error",
                    "tool": "compare_cim_values",
                    "cim_root": services["cim_root"],
                    "error": "missing_sv_values",
                    "answer": "Es konnten keine SV-Werte für den Vergleich ermittelt werden.",
                }

            if observed_mode == "abs_peak_timestamp":
                selected_observed = max(observed_rows, key=lambda item: abs(item["value"]))
            else:
                selected_observed = max(observed_rows, key=lambda item: item["value"])

            observed_value = selected_observed["value"]
            result_payload["observed_value"] = observed_value
            result_payload["observed_timestamp"] = selected_observed.get("timestamp")
        else:
            observed_values = _extract_observed_series(query_result_data)
            if not observed_values:
                return {
                    "status": "error",
                    "tool": "compare_cim_values",
                    "cim_root": services["cim_root"],
                    "error": "missing_sv_values",
                    "answer": "Es konnten keine SV-Werte für den Vergleich ermittelt werden.",
                }

            if observed_mode == "abs_peak":
                observed_value = max(abs(v) for v in observed_values)
            elif observed_mode == "mean":
                observed_value = sum(observed_values) / len(observed_values)
            else:
                observed_value = max(observed_values)

            result_payload["observed_value"] = observed_value

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
        observed_timestamp = result_payload.get("observed_timestamp")
        if observed_timestamp:
            answer = (
                f"Vergleich für {equipment_name}: höchster beobachteter Wert={observed_value:.3f} "
                f"am Zeitpunkt {observed_timestamp}, {limit_attr}={limit_value:.3f} -> {status_text} dem Grenzwert"
            )
        else:
            answer = (
                f"Vergleich für {equipment_name}: beobachteter Wert={observed_value:.3f}, "
                f"{limit_attr}={limit_value:.3f} -> {status_text} dem Grenzwert"
            )
    elif comparison_mode == "range_limit":
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
    elif comparison_mode == "range_limit_per_timestamp":
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

        violations: List[Dict[str, Any]] = []
        for item in observed_rows:
            point_value = item["value"]
            below = bool(lower_value is not None and point_value < lower_value)
            above = bool(upper_value is not None and point_value > upper_value)
            if below or above:
                violations.append({
                    "timestamp": item.get("timestamp"),
                    "observed_value": point_value,
                    "below_lower_limit": below,
                    "above_upper_limit": above,
                })

        within = len(violations) == 0
        result_payload.update({
            "lower_attribute": lower_attr,
            "lower_value": lower_value,
            "upper_attribute": upper_attr,
            "upper_value": upper_value,
            "within_limits": within,
            "num_points": len(observed_rows),
            "num_violations": len(violations),
            "violations": violations,
        })

        if within:
            answer = (
                f"Die zulässigen Grenzen für {equipment_name} wurden an allen {len(observed_rows)} verfügbaren Zeitpunkten eingehalten."
            )
        elif len(violations) == 1:
            violation = violations[0]
            ts = violation.get("timestamp") or "unbekannt"
            direction = "oberhalb der oberen Grenze" if violation.get("above_upper_limit") else "unterhalb der unteren Grenze"
            answer = (
                f"Die zulässigen Grenzen für {equipment_name} wurden nicht durchgehend eingehalten. "
                f"Am Zeitpunkt {ts} lag der beobachtete Wert bei {violation['observed_value']:.3f} und damit {direction}; "
                f"an den übrigen Zeitpunkten wurden die Grenzwerte eingehalten."
            )
        else:
            first_ts = violations[0].get("timestamp") or "unbekannt"
            answer = (
                f"Die zulässigen Grenzen für {equipment_name} wurden nicht durchgehend eingehalten. "
                f"Es gab {len(violations)} Verletzungen bei {len(observed_rows)} ausgewerteten Zeitpunkten; "
                f"die erste Verletzung trat am Zeitpunkt {first_ts} auf."
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

# lädt relevante Snapshots für Zeitfenster und erzeugt den Cache
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


# führt die eigentliche Domänenabfrage aus; Refactoring: Heuristik über llm_cim_orchestrator.handle_user_query(). _query_cim_with_services verwendet dafür handle_user_query
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



# formuliert die finale Antwort an den User 
def _build_final_answer_chain():
    parser = PydanticOutputParser(pydantic_object=FinalAnswerDecision)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the final answer formatter for a CIM domain agent.\n"
            "Your task is to produce the final German user-facing answer.\n"
            "Return only structured output.\n"
            "Use the raw execution result and align the answer tightly to the user's question.\n"
            "Do not invent values. Use only the provided execution data.\n"
            "If the user asked for utilization/loading (Auslastung), answer explicitly with the utilization semantics.\n"
            "If ratio_percent is available, prefer stating the utilization in percent.\n"
            "If the user asked whether something was within limits, answer explicitly yes/no in German and then add the supporting value.\n"
            "If the user asked for a comparison against a limit/base value, mention observed value and the relevant limit/base value.\n"
            "If the raw answer is already good, you may keep it concise but still adapt it to the user's wording.\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            "User request:\n{user_input}\n\n"
            "Raw answer:\n{raw_answer}\n\n"
            "Comparison resolution:\n{comparison_resolution}\n\n"
            "Comparison result:\n{comparison_result}\n\n"
            "Base values with units:\n{base_values_with_units}\n\n"
            "Query result data:\n{query_result_data}\n\n"
            "Parsed query:\n{parsed_query}\n\n"
            "Resolved equipment name:\n{equipment_name}"
        ),
    ])
    llm = get_llm()
    return prompt | llm | parser, parser

# nimmt den Roh-Output und lässt ihn LLM-seitig an die User-Frage angleichen
def _generate_user_aligned_answer(
    user_input: str,
    result_payload: Dict[str, Any],
) -> str:
    raw_answer = str(result_payload.get("answer", "") or "").strip()
    comparison_result = result_payload.get("comparison_result", {}) or {}
    comparison_resolution = result_payload.get("comparison_resolution", {}) or {}
    base_values_with_units = result_payload.get("base_values_with_units", {}) or {}
    query_result_data = result_payload.get("query_result_data", {}) or {}
    parsed_query = result_payload.get("parsed_query", {}) or {}
    resolved_object = result_payload.get("resolved_object")
    equipment_name = _safe_name(resolved_object) or getattr(resolved_object, "mRID", None) or "<unbekannt>"

    if not raw_answer:
        return raw_answer

    try:
        chain, parser = _build_final_answer_chain()
        decision = chain.invoke({
            "user_input": user_input,
            "raw_answer": raw_answer,
            "comparison_resolution": comparison_resolution,
            "comparison_result": comparison_result,
            "base_values_with_units": base_values_with_units,
            "query_result_data": query_result_data,
            "parsed_query": parsed_query,
            "equipment_name": equipment_name,
            "format_instructions": parser.get_format_instructions(),
        })
        answer = str(getattr(decision, "answer", "") or "").strip()
        return answer or raw_answer
    except Exception:
        return raw_answer

# baut den finalen Output und Debug-Block
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

    final_answer = _generate_user_aligned_answer(
        user_input=user_input,
        result_payload=result_payload,
    )

    return {
        "status": "ok",
        "tool": "summarize_cim_result",
        "cim_root": services["cim_root"],
        "user_input": user_input,
        "answer": final_answer,
        "debug": debug,
    }

# liest Low/High-Limits direkt von Node/Terminal-Pfaden
def _resolve_voltage_limits_from_node(node_obj: Any) -> Dict[str, Any]:
    lows = []
    highs = []

    limit_sets = []

    # Direct limits on node (important fix)
    direct_limit_sets = getattr(node_obj, "OperationalLimitSet", None) or []
    if not isinstance(direct_limit_sets, list):
        direct_limit_sets = [direct_limit_sets]
    limit_sets.extend(direct_limit_sets)

    # Limits via terminals
    terminals = getattr(node_obj, "Terminal", None) or getattr(node_obj, "Terminals", None) or []
    if not isinstance(terminals, list):
        terminals = [terminals]

    for t in terminals:
        term_limit_sets = getattr(t, "OperationalLimitSet", None) or []
        if not isinstance(term_limit_sets, list):
            term_limit_sets = [term_limit_sets]
        limit_sets.extend(term_limit_sets)

    for s in limit_sets:
        limits = getattr(s, "OperationalLimitValue", None) or getattr(s, "OperationalLimit", None) or []
        if not isinstance(limits, list):
            limits = [limits]

        for o in limits:
            if o is None:
                continue

            name = (getattr(o, "name", "") or "").lower()
            val = getattr(o, "value", None)
            if val is None:
                continue

            if "high" in name or "max" in name:
                highs.append(float(val))
            elif "low" in name or "min" in name:
                lows.append(float(val))

    return {
        "lowVoltageLimit": min(lows) if lows else None,
        "highVoltageLimit": max(highs) if highs else None,
    }
