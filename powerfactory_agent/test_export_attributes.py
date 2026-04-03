from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.pf_runner import _get_pf, _get_app, _activate_project_by_name


# Optional filter to reduce very noisy Python-side attributes from dir(obj)
EXCLUDED_DIR_NAMES = {
    "this",
    "thisown",
}


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        return []


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _safe_get_name(obj: Any) -> Optional[str]:
    return _safe_getattr(obj, "loc_name")


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


def _connect_powerfactory(project_name: str) -> Dict[str, Any]:
    pf = _get_pf()
    app = _get_app(pf)
    if app is None:
        raise RuntimeError("PowerFactory nicht erreichbar (GetApplication/GetApplicationExt ist None).")

    ok = _activate_project_by_name(app, project_name)
    if not ok:
        raise RuntimeError(f"Projekt konnte nicht aktiviert werden: {project_name}")

    project = app.GetActiveProject()
    if project is None:
        raise RuntimeError("Kein aktives Projekt.")

    studycase = app.GetActiveStudyCase()
    if studycase is None:
        raise RuntimeError("Kein aktiver Study Case.")

    return {
        "pf": pf,
        "app": app,
        "project": project,
        "studycase": studycase,
        "project_name": project_name,
    }


def _ensure_loadflow(studycase: Any) -> Dict[str, Any]:
    ldf_list = _to_list(studycase.GetContents("*.ComLdf", 1))
    if not ldf_list:
        ldf = studycase.CreateObject("ComLdf", "LoadFlow")
    else:
        ldf = ldf_list[0]

    rc = ldf.Execute()
    return {
        "status": "ok",
        "return_code": rc,
        "loadflow_name": _safe_getattr(ldf, "loc_name") or "LoadFlow",
    }


def _iter_unique_objects(app: Any, patterns: List[str]) -> List[Any]:
    seen: set[str] = set()
    unique: List[Any] = []
    for pattern in patterns:
        for obj in _to_list(app.GetCalcRelevantObjects(pattern)):
            key = _safe_get_full_name(obj) or _safe_get_name(obj)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(obj)
    unique.sort(key=lambda obj: ((_safe_get_name(obj) or ""), (_safe_get_full_name(obj) or "")))
    return unique


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
            for item in _to_list(result):
                _add(item)

    try:
        for name in dir(obj):
            if name.startswith("_") or name in EXCLUDED_DIR_NAMES:
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

    names.sort(key=str.lower)
    return names


def _safe_get_attribute(obj: Any, attr_name: str) -> Any:
    try:
        value = obj.GetAttribute(attr_name)
        if value is not None:
            return value
    except Exception:
        pass

    try:
        value = getattr(obj, attr_name)
        if value is not None:
            return value
    except Exception:
        pass

    return None


def _safe_get_attribute_description_text(obj: Any, attr_name: str) -> Optional[str]:
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
    for key in ["short", "short_text", "description", "text", "label", "name"]:
        try:
            value = getattr(desc, key, None)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            value_text = str(value).strip()
        except Exception:
            value_text = ""
        if value_text and value_text not in parts:
            parts.append(value_text)

    if parts:
        return " | ".join(parts)

    try:
        value_text = str(desc).strip()
    except Exception:
        value_text = ""
    return value_text or None


def _safe_get_attribute_unit(obj: Any, attr_name: str) -> Optional[str]:
    if obj is None or not attr_name:
        return None

    for method_name in ["GetAttributeUnit", "GetAttributeUnits"]:
        try:
            method = getattr(obj, method_name, None)
            if callable(method):
                unit = method(attr_name)
                if unit not in (None, ""):
                    return str(unit)
        except Exception:
            pass

    try:
        desc = obj.GetAttributeDescription(attr_name)
        unit = getattr(desc, "unit", None)
        if unit not in (None, ""):
            return str(unit)
    except Exception:
        pass

    for unit_attr in [f"{attr_name}:unit", f"{attr_name}.unit"]:
        try:
            unit = obj.GetAttribute(unit_attr)
            if unit not in (None, ""):
                return str(unit)
        except Exception:
            pass
        try:
            unit = getattr(obj, unit_attr)
            if unit not in (None, ""):
                return str(unit)
        except Exception:
            pass

    return None


def _coerce_scalar(value: Any) -> Tuple[Any, str]:
    if value is None:
        return None, "NoneType"
    if isinstance(value, (str, int, float, bool)):
        return value, type(value).__name__

    try:
        if hasattr(value, "__len__") and len(value) == 1:
            first = value[0]
            return first, type(first).__name__
    except Exception:
        pass

    return value, type(value).__name__


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        return repr(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return "<unprintable>"


def _read_attribute_record(obj: Any, attr_name: str) -> Dict[str, Any]:
    raw_value = _safe_get_attribute(obj, attr_name)
    scalar_value, value_type = _coerce_scalar(raw_value)
    description = _safe_get_attribute_description_text(obj, attr_name)
    unit = _safe_get_attribute_unit(obj, attr_name)
    readable = raw_value is not None

    return {
        "attribute_name": attr_name,
        "attribute_description": description,
        "value": scalar_value,
        "value_text": _stringify_value(scalar_value),
        "raw_value_text": _stringify_value(raw_value),
        "value_type": value_type,
        "unit": unit,
        "readable": readable,
        "data_source_hint": "result" if attr_name.startswith(("m:", "c:")) else "base",
    }


def _build_attribute_rows(
    app: Any,
    *,
    object_type: str,
    patterns: List[str],
    max_objects: Optional[int] = None,
    readable_only: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    objects = _iter_unique_objects(app, patterns)
    if max_objects is not None:
        objects = objects[: max(0, max_objects)]

    for obj in objects:
        object_info = {
            "object_type": object_type,
            "loc_name": _safe_get_name(obj),
            "full_name": _safe_get_full_name(obj),
            "pf_class": _safe_get_class_name(obj),
        }
        for attr_name in _discover_object_attribute_names(obj):
            record = _read_attribute_record(obj, attr_name)
            if readable_only and not record["readable"]:
                continue
            rows.append({**object_info, **record})
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    object_names = {(row.get("object_type"), row.get("full_name") or row.get("loc_name")) for row in rows}
    readable_count = sum(1 for row in rows if row.get("readable"))
    result_attr_count = sum(1 for row in rows if row.get("attribute_name", "").startswith(("m:", "c:")))
    return {
        "row_count": len(rows),
        "object_count": len(object_names),
        "readable_row_count": readable_count,
        "result_attr_row_count": result_attr_count,
    }


def export_all_loadflow_attributes_csv(
    project_name: str = DEFAULT_PROJECT_NAME,
    output_dir: str = "./pf_loadflow_all_attributes_export",
    max_bus_objects: Optional[int] = None,
    max_line_objects: Optional[int] = None,
    readable_only: bool = False,
) -> Dict[str, Any]:
    ctx = _connect_powerfactory(project_name)
    app = ctx["app"]
    studycase = ctx["studycase"]

    loadflow_info = _ensure_loadflow(studycase)

    bus_rows = _build_attribute_rows(
        app,
        object_type="bus",
        patterns=["*.ElmTerm"],
        max_objects=max_bus_objects,
        readable_only=readable_only,
    )
    line_rows = _build_attribute_rows(
        app,
        object_type="line",
        patterns=["*.ElmLne", "*.ElmCabl"],
        max_objects=max_line_objects,
        readable_only=readable_only,
    )

    output_path = Path(output_dir)
    buses_csv = output_path / "pf_loadflow_buses_all_attributes.csv"
    lines_csv = output_path / "pf_loadflow_lines_all_attributes.csv"

    _write_csv(buses_csv, bus_rows)
    _write_csv(lines_csv, line_rows)

    return {
        "status": "ok",
        "project": project_name,
        "studycase": _safe_getattr(studycase, "loc_name"),
        "loadflow": loadflow_info,
        "buses_csv": str(buses_csv.resolve()),
        "lines_csv": str(lines_csv.resolve()),
        "bus_summary": _build_summary(bus_rows),
        "line_summary": _build_summary(line_rows),
        "readable_only": readable_only,
        "max_bus_objects": max_bus_objects,
        "max_line_objects": max_line_objects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Führt einen Lastfluss aus und exportiert möglichst alle entdeckbaren Attribute "
            "von Bussen und Leitungen in Long-Format-CSV-Dateien."
        )
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT_NAME, help="PowerFactory-Projektname")
    parser.add_argument(
        "--output-dir",
        default="./pf_loadflow_all_attributes_export",
        help="Zielordner für die CSV-Dateien",
    )
    parser.add_argument(
        "--max-bus-objects",
        type=int,
        default=None,
        help="Optional: nur die ersten N Bus-Objekte exportieren",
    )
    parser.add_argument(
        "--max-line-objects",
        type=int,
        default=None,
        help="Optional: nur die ersten N Leitungs-/Kabelobjekte exportieren",
    )
    parser.add_argument(
        "--readable-only",
        action="store_true",
        help="Nur Attribute exportieren, deren Wert aktuell lesbar ist",
    )
    args = parser.parse_args()

    result = export_all_loadflow_attributes_csv(
        project_name=args.project,
        output_dir=args.output_dir,
        max_bus_objects=args.max_bus_objects,
        max_line_objects=args.max_line_objects,
        readable_only=args.readable_only,
    )

    print("[LOADFLOW ALL-ATTRIBUTES CSV EXPORT]")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
