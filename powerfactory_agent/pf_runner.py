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


def run_powerfactory_change(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    """
    Führt eine PowerFactory-Änderung aus (aktuell: 'Welche Last wie ändern') und rechnet Lastfluss vorher/nachher.
    """
    pf = _get_pf()

    # PowerFactory starten
    app = _get_app(pf)
    if app is None:
        return {
            "status": "error",
            "tool": "powerfactory",
            "error": "PowerFactory nicht erreichbar (GetApplication/GetApplicationExt ist None)",
        }

    # Projekt aktivieren (robust)
    ok = _activate_project_by_name(app, project_name)
    if not ok:
        return {
            "status": "error",
            "tool": "powerfactory",
            "error": f"Projekt konnte nicht aktiviert werden (nicht gefunden/kein Zugriff): {project_name}",
        }

    project = app.GetActiveProject()
    if project is None:
        return {
            "status": "error",
            "tool": "powerfactory",
            "error": "Projekt nicht aktiv (GetActiveProject() None)",
        }

    studycase = app.GetActiveStudyCase()
    if studycase is None:
        return {
            "status": "error",
            "tool": "powerfactory",
            "error": "Kein aktiver Study Case",
        }

    # Lastflusskommando holen/erstellen
    ldf_list = _to_py_list(studycase.GetContents("*.ComLdf", 1))
    if not ldf_list:
        ldf = studycase.CreateObject("ComLdf", "LoadFlow")
    else:
        ldf = ldf_list[0]

    # Agenten importieren
    # Fallback kompatibel für alte und neue Projektstruktur
    try:
        from cimpy.powerfactory_agent.agents.PowerFactoryAgent import PowerFactoryAgent
        from cimpy.powerfactory_agent.agents.LLM_interpreterAgent import LLM_interpreterAgent
        from cimpy.powerfactory_agent.agents.Result_interpreterAgent import Result_interpreterAgent
        from cimpy.powerfactory_agent.agents.LLM_resultAgent import LLM_resultAgent
    except ImportError:
        from cimpy.powerfactory_agent.agents.PowerFactoryAgent import PowerFactoryAgent
        from cimpy.powerfactory_agent.agents.LLM_interpreterAgent import LLM_interpreterAgent
        from cimpy.powerfactory_agent.agents.Result_interpreterAgent import Result_interpreterAgent
        from cimpy.powerfactory_agent.agents.LLM_resultAgent import LLM_resultAgent

    pf_agent = PowerFactoryAgent(project, studycase)
    llm_agent = LLM_interpreterAgent(project, studycase)
    result_agent = Result_interpreterAgent()
    llm_result_agent = LLM_resultAgent()

    # Eingabe interpretieren + Last auflösen
    instruction = llm_agent.interpret(user_input)
    print("[DEBUG] instruction:", instruction)

    if isinstance(instruction, dict) and "error" in instruction:
        return {
            "status": "error",
            "tool": "powerfactory",
            "error": instruction.get("error", "cannot_parse"),
            "details": instruction.get("details"),
        }

    resolved_load = llm_agent.resolve(instruction)

    # Lastfluss vor Änderung
    ldf.Execute()

    buses = app.GetCalcRelevantObjects("*.ElmTerm")
    u_before: Dict[str, float] = {}
    for bus in buses:
        u_before[bus.loc_name] = bus.GetAttribute("m:u")

    # Änderung ausführen (Last ändern)
    try:
        _ = resolved_load.GetAttribute("plini")
    except AttributeError:
        return {
            "status": "error",
            "tool": "powerfactory",
            "error": f"Last {getattr(resolved_load, 'loc_name', '<unknown>')} hat kein Attribut 'plini'",
        }

    pf_agent.execute(instruction, resolved_load)

    # Lastfluss nach Änderung
    ldf.Execute()

    u_after: Dict[str, float] = {}
    for bus in buses:
        u_after[bus.loc_name] = bus.GetAttribute("m:u")

    deltas: Dict[str, float] = {}
    for name, u0 in u_before.items():
        u1 = u_after.get(name)
        if u1 is not None:
            deltas[name] = u1 - u0

    # Faktische Interpretation + LLM Summary
    messages = result_agent.interpret_voltage_change(u_before, u_after)
    summary = llm_result_agent.summarize(messages, user_input)

    return {
        "status": "ok",
        "tool": "powerfactory",
        "project": project_name,
        "studycase": getattr(studycase, "loc_name", None),
        "instruction": instruction,
        "resolved_load": getattr(resolved_load, "loc_name", None),
        "data": {
            "u_before": u_before,
            "u_after": u_after,
            "delta_u": deltas,
        },
        "messages": messages,
        "answer": summary,
    }