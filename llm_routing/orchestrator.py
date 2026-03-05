from typing import Any, Dict, Optional

from cimpy.llm_routing.LLM_routeAgent import LLM_routeAgent
from cimpy.llm_routing.schemas import CallToolAction, AskUserAction, RouterAction
from cimpy.llm_routing.tools import historical_tool, powerfactory_tool


class Orchestrator:
    def __init__(self):
        self.router = LLM_routeAgent()
        self._pending: Optional[Dict[str, Any]] = None

    def handle(self, user_input: str) -> Dict[str, Any]:
        action: RouterAction = self.router.route(user_input, pending=self._pending) # Aufruf des LLM-Routers

        # falls LLM AskUserAction zurückgibt -> Ausgabe der gesammelten Infos für diesen Fall -> run_router stellt gezielte Nachfrage
        if isinstance(action, AskUserAction):
            self._pending = {
                "intended_tool": action.intended_tool,
                "missing_fields": action.missing_fields,
                "partial": action.partial,
            }
            return {
                "route": "ASK_USER",
                "question": action.question,
                "missing_fields": action.missing_fields,
                "partial": action.partial,
            }

        # falls LLM CallToolAction zurückgibt -> Ausgabe der gesammelten Infos für diesen Fall und Aufruf der Tools
        if isinstance(action, CallToolAction):
            self._pending = None # Historie wird gelöscht, da jetzt ausgeführt wird

            args = dict(action.args or {})
            args.setdefault("user_input", user_input)

            if action.tool == "historical":
                result = historical_tool.invoke(args)
            else:
                result = powerfactory_tool.invoke(args)

            return {"route": action.tool.upper(), "result": result}

        return {"route": "ERROR", "result": {"status": "error", "message": "Unknown action"}} # Fallback