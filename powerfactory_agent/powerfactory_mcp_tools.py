from __future__ import annotations

from typing import Any, Dict, Optional

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.pf_runner import _get_pf, _get_app, _activate_project_by_name, _to_py_list

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


# ------------------------------------------------------------------
# POWERFACTORY CONTEXT
# ------------------------------------------------------------------
def get_powerfactory_context(project_name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    pf = _get_pf()

    # PowerFactory starten
    app = _get_app(pf)
    if app is None:
        return {
            "status": "error",
            "tool": "powerfactory_context",
            "error": "PowerFactory nicht erreichbar (GetApplication/GetApplicationExt ist None)",
        }

    # Projekt aktivieren (robust)
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
# INTERNAL HELPERS USING EXISTING SERVICES
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


def _execute_change_load_with_services(services: Dict[str, Any], instruction: dict) -> Dict[str, Any]:
    app = services["app"]
    studycase = services["studycase"]
    interpreter = services["interpreter"]
    executor = services["executor"]
    project_name = services["project_name"]

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

    # Lastflusskommando holen/erstellen
    ldf_list = _to_py_list(studycase.GetContents("*.ComLdf", 1))
    if not ldf_list:
        ldf = studycase.CreateObject("ComLdf", "LoadFlow")
    else:
        ldf = ldf_list[0]

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
            "tool": "execute_change_load",
            "project": project_name,
            "instruction": instruction,
            "error": f"Last {getattr(resolved_load, 'loc_name', '<unknown>')} hat kein Attribut 'plini'",
        }

    execution_result = executor.execute(instruction, resolved_load)

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

    return {
        "status": "ok",
        "tool": "execute_change_load",
        "project": project_name,
        "studycase": getattr(studycase, "loc_name", None),
        "instruction": instruction,
        "resolved_load": getattr(resolved_load, "loc_name", None),
        "execution": execution_result,
        "data": {
            "u_before": u_before,
            "u_after": u_after,
            "delta_u": deltas,
        },
    }


def _summarize_powerfactory_result_with_services(
    services: Dict[str, Any],
    result_payload: dict,
    user_input: str,
) -> Dict[str, Any]:
    result_agent = services["result_agent"]
    llm_result_agent = services["llm_result_agent"]
    project_name = services["project_name"]

    data = result_payload.get("data", {}) if isinstance(result_payload, dict) else {}
    u_before = data.get("u_before", {}) if isinstance(data, dict) else {}
    u_after = data.get("u_after", {}) if isinstance(data, dict) else {}

    messages = result_agent.interpret_voltage_change(u_before, u_after)
    summary = llm_result_agent.summarize(messages, user_input)

    return {
        "status": "ok",
        "tool": "summarize_powerfactory_result",
        "project": project_name,
        "messages": messages,
        "answer": summary,
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
    return _summarize_powerfactory_result_with_services(
        services=services,
        result_payload=result_payload,
        user_input=user_input,
    )


# ------------------------------------------------------------------
# CONVENIENCE TOOL
# ------------------------------------------------------------------
def run_powerfactory_pipeline(
    user_input: str,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> Dict[str, Any]:
    services = build_powerfactory_services(project_name=project_name)
    if services["status"] != "ok":
        return services

    interpretation = _interpret_instruction_with_services(
        services=services,
        user_input=user_input,
    )
    if interpretation["status"] != "ok":
        return interpretation

    instruction = interpretation["instruction"]

    resolution = _resolve_load_with_services(
        services=services,
        instruction=instruction,
    )
    if resolution["status"] != "ok":
        return resolution

    execution = _execute_change_load_with_services(
        services=services,
        instruction=instruction,
    )
    if execution["status"] != "ok":
        return execution

    summary = _summarize_powerfactory_result_with_services(
        services=services,
        result_payload=execution,
        user_input=user_input,
    )
    if summary["status"] != "ok":
        return summary

    return {
        "status": "ok",
        "tool": "powerfactory",
        "project": project_name,
        "studycase": execution.get("studycase"),
        "instruction": instruction,
        "resolved_load": execution.get("resolved_load"),
        "data": execution.get("data", {}),
        "messages": summary.get("messages", []),
        "answer": summary.get("answer", ""),
        "debug": {
            "interpretation": interpretation,
            "resolution": resolution,
            "execution": execution,
            "summary": summary,
        },
    }