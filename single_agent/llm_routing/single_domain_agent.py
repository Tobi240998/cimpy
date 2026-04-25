from typing import Any, Dict, Optional

from cimpy.single_agent.llm_routing.LLM_routeAgent import LLM_routeAgent
from cimpy.single_agent.llm_routing.schemas import AskUserAction, CallToolAction, RouterAction

from cimpy.single_agent.pf.config import DEFAULT_PROJECT_NAME
from cimpy.single_agent.pf.powerfactory_domain_agent import PowerFactoryDomainAgent

from cimpy.llm_routing.config import CIM_ROOT
from cimpy.single_agent.cim.cim_domain_agent import CIMDomainAgent

from cimpy.single_agent.llm_routing.unified_plan import UnifiedPlan, UnifiedPlanStep
from cimpy.single_agent.llm_routing.unified_executor import UnifiedExecutor
from cimpy.single_agent.cim.cim_tool_registry import CIMToolRegistry
from cimpy.single_agent.cim.cim_planner import CIMPlanner

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
        self.router = LLM_routeAgent()
        self._pending: Optional[Dict[str, Any]] = None

        self.powerfactory_agent = PowerFactoryDomainAgent(project_name=project_name)
        self.cim_registry = CIMToolRegistry(cim_root=cim_root)
        self.cim_planner = CIMPlanner(
        cim_root=cim_root,
        registry=self.cim_registry,
    )

        self.executor = UnifiedExecutor(
            cim_registry=self.cim_registry,
            cim_root=cim_root,
            powerfactory_agent=self.powerfactory_agent,
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        action: RouterAction = self.router.route(user_input, pending=self._pending)

        if isinstance(action, AskUserAction):
            self._pending = {
                "intended_tool": action.intended_tool,
                "missing_fields": action.missing_fields,
                "partial": {
                    **(action.partial or {}),
                    "user_input": (action.partial or {}).get("user_input", user_input),
                },
                "question": action.question,
            }

            return {
                "route": "ASK_USER",
                "status": "needs_clarification",
                "question": action.question,
                "missing_fields": action.missing_fields,
                "partial": action.partial,
            }

        if isinstance(action, CallToolAction):
            pending = self._pending
            args = dict(action.args or {})

            original_user_input = None
            if pending:
                partial = pending.get("partial") or {}
                original_user_input = partial.get("user_input")

            resolved_user_input = args.get("user_input") or original_user_input or user_input
            self._pending = None

            if action.tool == "historical":
                plan = self._build_cim_unified_plan(resolved_user_input)
                result = self.executor.execute(plan)
                return {
                    "route": "CIM",
                    "domain": "cim",
                    "result": result,
                    "answer": result.get("answer"),
                }

            if action.tool == "powerfactory":
                plan = self._build_powerfactory_unified_plan(resolved_user_input)
                result = self.executor.execute(plan)
                return {
                    "route": "POWERFACTORY",
                    "domain": "powerfactory",
                    "result": result,
                }

            return {
                "route": "ERROR",
                "status": "error",
                "result": {
                    "status": "error",
                    "message": f"Unknown domain/tool: {action.tool}",
                },
            }

        return {
            "route": "ERROR",
            "status": "error",
            "result": {
                "status": "error",
                "message": "Unknown router action",
            },
        }
    
    def build_domain_plan(self, domain: str, user_input: str) -> UnifiedPlan:
        if domain == "powerfactory":
            return self.build_powerfactory_plan(user_input)

        if domain == "cim":
            return self.build_cim_plan(user_input)

        raise ValueError(f"Unknown domain: {domain}")

        def build_domain_plan(self, domain: str, user_input: str) -> UnifiedPlan:
            if domain == "powerfactory":
                return self._build_powerfactory_unified_plan(user_input)

            if domain == "cim":
                return self._build_cim_unified_plan(user_input)

            raise ValueError(f"Unknown domain: {domain}")

    def _build_powerfactory_unified_plan(self, user_input: str) -> UnifiedPlan:
        classification = self.powerfactory_agent.classify_request(user_input)
        raw_plan = self.powerfactory_agent.build_plan(
            user_input=user_input,
            classification=classification,
        )

        steps = [
            UnifiedPlanStep(
                domain="powerfactory",
                tool=item.get("step", ""),
                description=item.get("description", ""),
                mutating=bool(item.get("mutating", False)),
                args={
                    "user_input_override": item.get("user_input_override"),
                    "source_subrequest": item.get("source_subrequest"),
                },
            )
            for item in raw_plan
        ]

        return UnifiedPlan(
            domain="powerfactory",
            user_input=user_input,
            steps=steps,
            classification=classification,
            reasoning=self.powerfactory_agent._last_planning_debug.get("plan_source", ""),
        )

    def _build_cim_unified_plan(self, user_input: str) -> UnifiedPlan:
        classification = self.cim_planner.classify_request(user_input)
        raw_plan = self.cim_planner.build_plan(classification)

        steps = [
            UnifiedPlanStep(
                domain="cim",
                tool=item.get("step", ""),
                description=item.get("description", ""),
                mutating=bool(item.get("mutating", False)),
            )
            for item in raw_plan
        ]

        return UnifiedPlan(
            domain="cim",
            user_input=user_input,
            steps=steps,
            classification=classification,
            reasoning=classification.get("reasoning", ""),
        )