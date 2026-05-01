from typing import Any, Dict, Optional

from cimpy.single_agent2.llm_routing.unified_executor import UnifiedExecutor
from cimpy.single_agent2.llm_routing.unified_planner import UnifiedPlanner
from cimpy.single_agent2.llm_routing.unified_tool_registry import UnifiedToolRegistry
from cimpy.single_agent2.llm_routing.config import CIM_ROOT

from cimpy.single_agent2.pf.config import DEFAULT_PROJECT_NAME
from cimpy.single_agent2.pf.powerfactory_tool_registry import PowerFactoryToolRegistry
from cimpy.single_agent2.cim.cim_tool_registry import CIMToolRegistry

class SingleDomainAgent:
    """
    Phase-1 Single Agent:
    - entscheidet intern zwischen CIM und PowerFactory
    - nutzt vorerst die bestehenden DomainAgents weiter
    - hält pending clarification state selbst
    """

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT_NAME,
        cim_root: str = CIM_ROOT,
        debug_mode: bool = True,
    ):
        self.debug_mode = debug_mode
        self._pending: Optional[Dict[str, Any]] = None

        self.cim_registry = CIMToolRegistry(cim_root=cim_root)
        self.pf_registry = PowerFactoryToolRegistry()

        self.registry = UnifiedToolRegistry(
            cim_registry=self.cim_registry,
            pf_registry=self.pf_registry,
        )

        self.planner = UnifiedPlanner(
            registry=self.registry,
        )

        self.executor = UnifiedExecutor(
            cim_registry=self.cim_registry,
            cim_root=cim_root,
            pf_registry=self.pf_registry,
            project_name=project_name,
        )


    def _debug(self, label: str, value: Any = None) -> None:
        if not self.debug_mode:
            return

        print(f"\n[DEBUG SingleAgent2] {label}")
        if value is not None:
            try:
                import json
                print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
            except Exception:
                print(value)

    def run(self, user_input: str) -> Dict[str, Any]:
        
        self._debug("USER_INPUT", user_input)
        
        composite = self.planner.decompose_request(user_input)
        self._debug("COMPOSITE_DECISION", composite)

        if composite.get("is_composite"):
            return self._run_composite(user_input, composite)

        return self._run_single(user_input)
    
    def _run_single(self, user_input: str) -> Dict[str, Any]:
        planner_result = self.planner.build_plan(user_input)
        self._debug("PLANNER_RESULT", planner_result)

        if planner_result.get("status") == "needs_clarification":
            self._pending = {
                "question": planner_result.get("question"),
                "missing_context": planner_result.get("missing_context", []),
                "partial": {"user_input": user_input},
                "planner_decision": planner_result.get("planner_decision", {}),
            }

            return {
                "route": "ASK_USER",
                "status": "needs_clarification",
                "question": planner_result.get("question"),
                "missing_fields": planner_result.get("missing_context", []),
                "partial": {"user_input": user_input},
            }

        if planner_result.get("status") != "ok":
            return {
                "route": "ERROR",
                "status": "error",
                "result": planner_result,
            }

        self._pending = None
        plan = planner_result["plan"]
        self._debug(
            "UNIFIED_PLAN",
            plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
        )
        result = self.executor.execute(plan)
        debug = result.get("debug", {}) if isinstance(result, dict) else {}
        trace = debug.get("trace", []) if isinstance(debug, dict) else []

        self._debug("EXECUTED_STEPS", [
            {
                "step": item.get("step"),
                "status": (item.get("result") or {}).get("status")
                if isinstance(item.get("result"), dict) else None,
                "error": (item.get("result") or {}).get("error")
                if isinstance(item.get("result"), dict) else None,
                "details": (item.get("result") or {}).get("details")
                if isinstance(item.get("result"), dict) else None,
            }
            for item in trace
            if isinstance(item, dict)
        ])

        self._debug("EXECUTION_RESULT", result)

        return {
            "route": plan.domain.upper(),
            "domain": plan.domain,
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
            "planner_decision": planner_result.get("planner_decision"),
            "result": result,
        }
    
    def _run_composite(self, original_user_input: str, composite: Dict[str, Any]) -> Dict[str, Any]:
        subrequests = composite.get("subrequests", []) or []

        results = []
        combined_answers = []
        previous_context = ""

        for idx, subrequest in enumerate(subrequests, start=1):
            if isinstance(subrequest, dict):
                sub_input = subrequest.get("user_input", "")
                depends_on_previous = bool(subrequest.get("depends_on_previous", False))
            else:
                sub_input = getattr(subrequest, "user_input", "")
                depends_on_previous = bool(getattr(subrequest, "depends_on_previous", False))

            if not sub_input:
                continue

            effective_input = sub_input
            if depends_on_previous and previous_context:
                effective_input = (
                    f"Originale Gesamtanfrage: {original_user_input}\n\n"
                    f"Vorheriger Kontext:\n{previous_context}\n\n"
                    f"Aktuelle Teilanfrage:\n{sub_input}"
                )

            single_result = self._run_single(effective_input)

            results.append({
                "index": idx,
                "user_input": sub_input,
                "effective_input": effective_input,
                "depends_on_previous": depends_on_previous,
                "output": single_result,
            })

            answer = self._extract_answer_from_output(single_result)
            if answer:
                combined_answers.append(f"{idx}. {answer}")
                previous_context += f"\nTeil {idx}: {answer}"

            if single_result.get("route") == "ASK_USER":
                return {
                    "route": "ASK_USER",
                    "status": "needs_clarification",
                    "question": single_result.get("question"),
                    "missing_fields": single_result.get("missing_fields", []),
                    "partial": {
                        "user_input": original_user_input,
                        "composite": composite,
                        "completed_results": results,
                    },
                }

            if single_result.get("route") == "ERROR":
                return {
                    "route": "ERROR",
                    "status": "error",
                    "result": {
                        "status": "error",
                        "answer": "Die zusammengesetzte Anfrage konnte nicht vollständig ausgeführt werden.",
                        "composite": composite,
                        "subresults": results,
                    },
                }

        answer = "\n".join(combined_answers)

        return {
            "route": "COMPOSITE",
            "domain": "composite",
            "status": "ok",
            "is_composite": True,
            "composite_decision": composite,
            "result": {
                "status": "ok",
                "answer": answer,
                "subresults": results,
            },
        }
    
    def _extract_answer_from_output(self, out: Dict[str, Any]) -> str:
        if not isinstance(out, dict):
            return ""

        if out.get("route") == "ASK_USER":
            return str(out.get("question", ""))

        result = out.get("result", {})
        if isinstance(result, dict):
            answer = result.get("answer")
            if isinstance(answer, str):
                return answer

        return ""




    