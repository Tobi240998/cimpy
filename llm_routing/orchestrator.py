from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate

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
                result = powerfactory_tool.invoke(args)

            final_answer = build_final_answer_with_llm(
                llm=self.router.llm,
                user_input=args.get("user_input", user_input),
                route=action.tool,
                execution_result=result,
            )

            # Robust gegen Nicht-Dict-Ergebnisse
            if not isinstance(result, dict):
                result = {"answer": str(result)}

            # Rohantwort sichern
            result["raw_answer"] = result.get("answer")

            # Finale LLM-Antwort normalisieren
            if hasattr(final_answer, "content"):
                result["answer"] = final_answer.content
            else:
                result["answer"] = str(final_answer)

            return {
                "route": action.tool.upper(),
                "result": result,
            }

        return {"route": "ERROR", "result": {"status": "error", "message": "Unknown action"}} # Fallback
    

def build_final_answer_with_llm(
    llm,
    user_input: str,
    route: str,
    execution_result: dict,
) -> str:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are the final answer formatter of an energy-grid analysis agent.

Your task:
- convert the provided execution result into a natural user-facing answer
- use only information explicitly present in the execution result
- preserve all numeric values, object names, timestamps, units, and statuses exactly
- do not infer missing values
- do not perform calculations
- do not add explanations that are not present in the execution result
- do not diagnose the system unless the execution result contains an error
- if execution failed, state the error reason clearly and briefly
- if the execution result already contains an answer, prefer that answer and only improve wording minimally
- return only the final answer text
"""
        ),
        (
            "user",
            """
Original user request:
{user_input}

Route/domain:
{route}

Execution result:
{execution_result}
"""
        )
    ])

    chain = prompt | llm
    result = chain.invoke({
        "user_input": user_input,
        "route": route,
        "execution_result": execution_result,
    })

    return result.content if hasattr(result, "content") else str(result)