from typing import Any, Dict, Optional

from cimpy.llm_routing.LLM_routeAgent import LLM_routeAgent
from cimpy.llm_routing.schemas import CallToolAction, AskUserAction, RouterAction
from cimpy.llm_routing.tools import historical_tool
from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.powerfactory_domain_agent import PowerFactoryDomainAgent


class Orchestrator:
    def __init__(self):
        self.router = LLM_routeAgent()
        self._pending: Optional[Dict[str, Any]] = None
        self.powerfactory_agent = PowerFactoryDomainAgent(project_name=DEFAULT_PROJECT_NAME)

    def handle(self, user_input: str) -> Dict[str, Any]:
        action: RouterAction = self.router.route(user_input, pending=self._pending) # Aufruf des LLM-Routers

        # falls LLM AskUserAction zurückgibt -> Ausgabe der gesammelten Infos für diesen Fall -> run_router stellt gezielte Nachfrage
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
                "question": action.question,
                "missing_fields": action.missing_fields,
                "partial": action.partial,
            }

        # falls LLM CallToolAction zurückgibt -> Ausgabe der gesammelten Infos für diesen Fall und Aufruf der Tools
        if isinstance(action, CallToolAction):
            pending = self._pending

            args = dict(action.args or {})

            original_user_input = None
            if pending:
                partial = pending.get("partial") or {}
                original_user_input = partial.get("user_input")

            args.setdefault("user_input", original_user_input or user_input)

            self._pending = None

            if action.tool == "historical":
                result = historical_tool.invoke(args)
            else:
                result = self.powerfactory_agent.run(args.get("user_input", user_input))

            return {"route": action.tool.upper(), "result": result}

        return {"route": "ERROR", "result": {"status": "error", "message": "Unknown action"}} # Fallback