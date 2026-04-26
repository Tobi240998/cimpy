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
    ):
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

    def run(self, user_input: str) -> Dict[str, Any]:
        planner_result = self.planner.build_plan(user_input)

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
        result = self.executor.execute(plan)

        return {
            "route": plan.domain.upper(),
            "domain": plan.domain,
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
            "planner_decision": planner_result.get("planner_decision"),
            "result": result,
        }




    