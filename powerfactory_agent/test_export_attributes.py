from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openpyxl import Workbook
    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.pf_runner import _get_pf, _get_app, _activate_project_by_name


OBJECT_PATTERNS: Dict[str, List[str]] = {
    "bus": ["*.ElmTerm"],
    "line": ["*.ElmLne", "*.ElmCabl", "*.ElmCable"],
    "load": ["*.ElmLod*"],
    "transformer": ["*.ElmTr*"],
    "generator": ["*.ElmSym", "*.ElmAsm", "*.ElmGenstat", "*.ElmPvsys", "*.ElmSgen"],
    "switch": ["*.Sta*", "*.ElmCoup*", "*.RelFuse*"],
}


RESULT_PREFIXES = ("m:", "c:", "n:", "s:")
BASE_HINT_PREFIXES = ("e:", "b:")

# Deliberately conservative helper-field filter.
# Goal: keep potentially useful engineering data, filter obvious internal/technical helper fields.
TECHNICAL_HELPER_NAME_PATTERNS = [
    "loc_name",
    "for_name",
    "fold_id",
    "cpgrid",
    "cpzone",
    "cparea",
    "cpsubstat",
    "desc",
    "idetail",
    "ishclne",
    "gps",
    "gco",
    "chr",
    "colour",
    "color",
    "sym",
    "diagram",
    "display",
    "plot",
    "vis",
    "icon",
    "grf",
    "graphic",
    "intmon",
    "outserv",
    "isclosed",
    "on_off",
    "ukid",
    "busid",
    "fold",
    "frnom",
    "ausage",
    "iusage",
    "cpfeed",
    "cptrf",
    "cpterm",
    "cpnode",
    "cpobj",
    "typ_id",
    "type_id",
]

TECHNICAL_HELPER_LABEL_PATTERNS = [
    "access time",
    "locacctime",
    "accesstime",
    "graphic",
    "diagram",
    "layout",
    "display",
    "icon",
    "colour",
    "color",
    "usage",
    "identifier",
    "internal",
    "folder",
    "database",
]

ENGINEERING_LABEL_PATTERNS = [
    "spannung",
    "voltage",
    "strom",
    "current",
    "leistung",
    "power",
    "blindleistung",
    "wirkleistung",
    "loading",
    "auslastung",
    "nenn",
    "rated",
    "impedance",
    "resistance",
    "reactance",
    "admittance",
    "frequency",
    "length",
    "länge",
    "tap",
    "ratio",
    "setpoint",
    "soll",
    "limit",
    "grenze",
    "phase",
    "earth",
    "ground",
    "r0",
    "x0",
    "r1",
    "x1",
    "sn",
    "uknom",
    "inom",
]


def _to_py_list(value: Any) -> List[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


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
        if not name_str or name_str.startswith("_") or name_str in seen:
            return
        seen.add(name_str)
        names.append(name_str)

    for method_name in ["GetAttributeList", "GetAttributeNames", "GetVariableList", "GetVarList"]:
        method = getattr(obj, method_name, None)
        if not callable(method):
            continue
        for args in [(), ("*",), (0,), (1,)]:
            try:
                result = method(*args)
            except Exception:
                continue
            for item in _to_py_list(result):
                _add(item)

    try:
        for name in dir(obj):
            if name.startswith("_"):
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

    names.sort()
    return names


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        pass
    try:
        if hasattr(value, "__len__") and len(value) == 1:
            return float(value[0])
    except Exception:
        pass
    return None


def _get_pf_attribute_unit(obj: Any, attr_name: str) -> Optional[str]:
    if not obj or not attr_name:
        return None

    for method_name in ["GetAttributeUnit", "GetAttributeUnits"]:
        try:
            method = getattr(obj, method_name)
            unit = method(attr_name)
            if unit:
                return str(unit)
        except Exception:
            pass

    for method_name in ["GetAttributeDescription", "GetAttributeInfo", "GetVarInfo"]:
        try:
            method = getattr(obj, method_name)
            desc = method(attr_name)
            if desc is None:
                continue
            for field_name in ["unit", "Unit", "sUnit"]:
                try:
                    value = getattr(desc, field_name)
                    if value:
                        return str(value)
                except Exception:
                    pass
            if isinstance(desc, dict):
                for field_name in ["unit", "Unit", "sUnit"]:
                    value = desc.get(field_name)
                    if value:
                        return str(value)
        except Exception:
            pass

    try:
        value = obj.GetAttribute(f"{attr_name}:unit")
        if value:
            return str(value)
    except Exception:
        pass

    return None


def _extract_text_from_pf_metadata(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ["label", "name", "description", "desc", "short", "text", "title"]:
            item = value.get(key)
            if item is not None:
                text = str(item).strip()
                if text:
                    return text
        return None
    for attr in ["label", "name", "description", "desc", "short", "text", "title"]:
        try:
            item = getattr(value, attr)
            if item is not None:
                text = str(item).strip()
                if text:
                    return text
        except Exception:
            pass
    try:
        text = str(value).strip()
        return text or None
    except Exception:
        return None


def _get_pf_attribute_label(obj: Any, attr_name: str) -> Optional[str]:
    if not obj or not attr_name:
        return None

    for method_name in ["GetAttributeLabel", "GetAttributeName", "GetVarName"]:
        try:
            method = getattr(obj, method_name)
            value = method(attr_name)
            text = _extract_text_from_pf_metadata(value)
            if text:
                return text
        except Exception:
            pass

    for method_name in ["GetAttributeDescription", "GetAttributeInfo", "GetVarInfo"]:
        try:
            method = getattr(obj, method_name)
            desc = method(attr_name)
            if desc is None:
                continue
            for field_name in ["label", "Label", "name", "Name", "short", "Short", "title", "Title"]:
                try:
                    value = getattr(desc, field_name)
                    text = _extract_text_from_pf_metadata(value)
                    if text:
                        return text
                except Exception:
                    pass
            if isinstance(desc, dict):
                for field_name in ["label", "Label", "name", "Name", "short", "Short", "title", "Title"]:
                    text = _extract_text_from_pf_metadata(desc.get(field_name))
                    if text:
                        return text
        except Exception:
            pass

    return None


def _get_pf_attribute_description(obj: Any, attr_name: str) -> Optional[str]:
    if not obj or not attr_name:
        return None

    for method_name in ["GetAttributeDescription", "GetAttributeInfo", "GetVarInfo"]:
        try:
            method = getattr(obj, method_name)
            desc = method(attr_name)
            if desc is None:
                continue

            if isinstance(desc, str):
                text = desc.strip()
                if text:
                    return text

            for field_name in ["description", "Description", "desc", "Desc", "help", "Help", "text", "Text"]:
                try:
                    value = getattr(desc, field_name)
                    text = _extract_text_from_pf_metadata(value)
                    if text:
                        return text
                except Exception:
                    pass

            if isinstance(desc, dict):
                for field_name in ["description", "Description", "desc", "Desc", "help", "Help", "text", "Text"]:
                    text = _extract_text_from_pf_metadata(desc.get(field_name))
                    if text:
                        return text
        except Exception:
            pass

    return None


def _read_attribute(obj: Any, attr_name: str) -> Dict[str, Any]:
    raw_value = None
    read_ok = False

    try:
        raw_value = obj.GetAttribute(attr_name)
        read_ok = raw_value is not None
    except Exception:
        raw_value = None

    if raw_value is None:
        try:
            raw_value = getattr(obj, attr_name)
            read_ok = raw_value is not None
        except Exception:
            raw_value = None

    numeric_value = _coerce_numeric(raw_value)
    pf_unit = _get_pf_attribute_unit(obj, attr_name)
    pf_label = _get_pf_attribute_label(obj, attr_name)
    pf_description = _get_pf_attribute_description(obj, attr_name)

    return {
        "read_ok": read_ok,
        "raw_value": raw_value,
        "numeric_value": numeric_value,
        "display_value": numeric_value if numeric_value is not None else raw_value,
        "pf_unit": pf_unit,
        "unit_source": "powerfactory" if pf_unit else None,
        "pf_label": pf_label,
        "pf_description": pf_description,
        "value_type": type(raw_value).__name__ if raw_value is not None else None,
    }


def _classify_data_origin(attr_name: str) -> str:
    lower = (attr_name or "").strip().lower()
    if lower.startswith(RESULT_PREFIXES):
        return "result"
    if lower.startswith(BASE_HINT_PREFIXES):
        return "base"
    return "base_or_parameter"


def _contains_any(text: str, patterns: List[str]) -> bool:
    lower = (text or "").strip().lower()
    return any(pattern in lower for pattern in patterns)


def _classify_attribute_for_queryability(
    attr_name: str,
    pf_label: Optional[str],
    pf_description: Optional[str],
    read_ok: bool,
    pf_unit: Optional[str],
    value_type: Optional[str],
) -> Tuple[str, str]:
    """
    Returns (queryability_filter, filter_reason)

    queryability_filter:
      - keep
      - review
      - helper_filtered
    """
    attr_lower = (attr_name or "").strip().lower()
    label_lower = (pf_label or "").strip().lower()
    desc_lower = (pf_description or "").strip().lower()
    merged_text = " | ".join(x for x in [attr_lower, label_lower, desc_lower] if x)

    if _contains_any(attr_lower, TECHNICAL_HELPER_NAME_PATTERNS):
        return "helper_filtered", "attr_name_matches_helper_pattern"

    if _contains_any(label_lower, TECHNICAL_HELPER_LABEL_PATTERNS) or _contains_any(desc_lower, TECHNICAL_HELPER_LABEL_PATTERNS):
        return "helper_filtered", "pf_metadata_matches_helper_pattern"

    if pf_unit:
        return "keep", "pf_unit_available"

    if _classify_data_origin(attr_name) == "result":
        return "keep", "result_attribute_prefix"

    if _contains_any(merged_text, ENGINEERING_LABEL_PATTERNS):
        if read_ok:
            return "keep", "engineering_metadata_or_name_match"
        return "review", "engineering_match_but_not_readable"

    if read_ok and value_type in {"int", "float", "list", "tuple"}:
        return "review", "readable_but_no_pf_unit_or_clear_metadata"

    if read_ok:
        return "review", "readable_textual_or_untyped_attribute"

    return "helper_filtered", "not_readable_and_no_engineering_signal"


def _collect_objects(app: Any, patterns: List[str]) -> List[Any]:
    objects: List[Any] = []
    seen = set()
    for pattern in patterns:
        try:
            found = app.GetCalcRelevantObjects(pattern) or []
        except Exception:
            found = []
        for obj in found:
            key = _safe_get_full_name(obj) or _safe_get_name(obj)
            if not key or key in seen:
                continue
            seen.add(key)
            objects.append(obj)
    return objects


def export_attributes(
    project_name: str = DEFAULT_PROJECT_NAME,
    output_dir: str = ".",
    max_objects_per_type: Optional[int] = None,
    only_readable: bool = False,
) -> Dict[str, Any]:
    pf = _get_pf()
    app = _get_app(pf)
    if app is None:
        raise RuntimeError("PowerFactory nicht erreichbar (GetApplication/GetApplicationExt ist None).")

    ok = _activate_project_by_name(app, project_name)
    if not ok:
        raise RuntimeError(f"Projekt konnte nicht aktiviert werden: {project_name}")

    project = app.GetActiveProject()
    studycase = app.GetActiveStudyCase()
    if project is None:
        raise RuntimeError("Kein aktives Projekt.")
    if studycase is None:
        raise RuntimeError("Kein aktiver Study Case.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for entity_type, patterns in OBJECT_PATTERNS.items():
        objects = _collect_objects(app, patterns)
        if max_objects_per_type is not None:
            objects = objects[:max_objects_per_type]

        for obj in objects:
            object_name = _safe_get_name(obj)
            full_name = _safe_get_full_name(obj)
            pf_class = _safe_get_class_name(obj)
            attr_names = _discover_object_attribute_names(obj)

            for attr_name in attr_names:
                read_result = _read_attribute(obj, attr_name)

                if only_readable and not read_result["read_ok"]:
                    continue

                queryability_filter, filter_reason = _classify_attribute_for_queryability(
                    attr_name=attr_name,
                    pf_label=read_result["pf_label"],
                    pf_description=read_result["pf_description"],
                    read_ok=read_result["read_ok"],
                    pf_unit=read_result["pf_unit"],
                    value_type=read_result["value_type"],
                )

                rows.append({
                    "entity_type": entity_type,
                    "pf_class": pf_class,
                    "object_name": object_name,
                    "full_name": full_name,
                    "attribute_name": attr_name,
                    "attribute_label": read_result["pf_label"],
                    "attribute_description": read_result["pf_description"],
                    "data_origin": _classify_data_origin(attr_name),
                    "read_ok": read_result["read_ok"],
                    "display_value": read_result["display_value"],
                    "raw_value_repr": repr(read_result["raw_value"]) if read_result["raw_value"] is not None else None,
                    "numeric_value": read_result["numeric_value"],
                    "unit": read_result["pf_unit"],
                    "unit_source": read_result["unit_source"],
                    "value_type": read_result["value_type"],
                    "queryability_filter": queryability_filter,
                    "filter_reason": filter_reason,
                })

    csv_path = output_path / "powerfactory_attribute_export.csv"
    fieldnames = [
        "entity_type",
        "pf_class",
        "object_name",
        "full_name",
        "attribute_name",
        "attribute_label",
        "attribute_description",
        "data_origin",
        "read_ok",
        "display_value",
        "raw_value_repr",
        "numeric_value",
        "unit",
        "unit_source",
        "value_type",
        "queryability_filter",
        "filter_reason",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    xlsx_path = None
    if HAS_OPENPYXL:
        wb = Workbook()
        ws = wb.active
        ws.title = "PF Attributes"
        ws.append(fieldnames)
        for row in rows:
            ws.append([row.get(col) for col in fieldnames])
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(value))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 80)
        xlsx_path = output_path / "powerfactory_attribute_export.xlsx"
        wb.save(xlsx_path)

    summary_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        entity_type = row["entity_type"]
        filter_name = row["queryability_filter"]
        summary_counts.setdefault(entity_type, {})
        summary_counts[entity_type][filter_name] = summary_counts[entity_type].get(filter_name, 0) + 1

    return {
        "status": "ok",
        "project_name": project_name,
        "studycase": getattr(studycase, "loc_name", None),
        "rows": len(rows),
        "csv_path": str(csv_path),
        "xlsx_path": str(xlsx_path) if xlsx_path else None,
        "summary_counts": summary_counts,
    }


if __name__ == "__main__":
    result = export_attributes(
        project_name=DEFAULT_PROJECT_NAME,
        output_dir=".",
        max_objects_per_type=None,
        only_readable=False,
    )
    print(result)
