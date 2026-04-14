from __future__ import annotations

from typing import Any, Dict
import sys

from cimpy.powerfactory_agent.config import PF_PYTHON_PATH, DEFAULT_PROJECT_NAME


def _get_pf():
    # Import zur Laufzeit, damit Import des Moduls nicht sofort PF verlangt.
    if PF_PYTHON_PATH not in sys.path:
        sys.path.append(PF_PYTHON_PATH)
    import powerfactory as pf  # type: ignore
    return pf


def _to_py_list(obj):
    """
    Macht aus typischen PowerFactory Rückgaben eine echte Python-Liste.
    (GetContents(...) kann je nach Version list / [list] / DataObject liefern)
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        if len(obj) == 1 and isinstance(obj[0], list):
            return obj[0]
        return obj
    try:
        return list(obj)
    except Exception:
        pass
    try:
        children = obj.GetChildren()
        if children is None:
            return []
        if isinstance(children, list):
            return children
        return list(children)
    except Exception:
        return []


def _get_app(pf):
    app = pf.GetApplication()  # versucht vorhandene Instanz
    if app is None:
        app = pf.GetApplicationExt()  # startet neue
    return app


def _activate_project_by_name(app, project_name: str) -> bool:
    """
    Robuste Projektaktivierung über CurrentUser -> *.IntPrj -> Activate().
    (app.ActivateProject(name) ist je nach Setup/Version unzuverlässig)
    """
    user = app.GetCurrentUser()
    if user is None:
        return False

    raw = user.GetContents("*.IntPrj", 0)
    projects = _to_py_list(raw)
    for p in projects:
        if getattr(p, "loc_name", None) == project_name:
            try:
                p.Activate()
                return True
            except Exception:
                return False
    return False


