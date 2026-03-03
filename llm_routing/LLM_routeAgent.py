import json
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from cimpy.llm_routing.langchain_llm import get_llm
from cimpy.llm_routing.schemas import RouterAction, CallToolAction, AskUserAction


class LLM_routeAgent:
    def __init__(self):
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist ein Routing-Controller für zwei Tools: historical und powerfactory.\n\n"
            "Du MUSST ausschließlich gültiges JSON ausgeben und sonst nichts.\n\n"
            "Erlaubte Outputs:\n"
            "1) Tool Call:\n"
            '{ "action": "call_tool", "tool": "historical|powerfactory", "args": { ... } }\n\n'
            "2) Rückfrage (wenn Pflichtinfos fehlen):\n"
            '{ "action": "ask_user", "question": "...", "missing_fields": ["..."], "partial": { ... }, "intended_tool": "historical|powerfactory" }\n\n'
            "Routing-Regeln:\n"
            "- Vergangenheit, Verlauf, Durchschnitt, Maximum, Datum, 'war', 'über den Tag' -> historical.\n"
            "- Änderungen, Schalten, Last ändern, 'rechne Lastfluss', 'setze', 'ändere' -> powerfactory.\n\n"
            "Args-Minimum:\n"
            "- args muss mindestens {\"user_input\": <Nutzertext>} enthalten.\n\n"
            "Wenn für historical der Zeitraum unklar ist (z.B. 'über den Tag' ohne Angabe wie 'heute/gestern/Datum'), "
            "stelle eine Rückfrage und setze missing_fields=['time_range'].\n"
            "time_range darf ein einfacher String sein (z.B. 'gestern', '2026-03-02')."
        )

    def route(self, user_input: str, pending: Optional[Dict[str, Any]] = None) -> RouterAction:
        context = ""
        if pending:
            context = (
                "\n\nKONTEXT (laufende Klärung, berücksichtigen!):\n"
                + json.dumps(pending, ensure_ascii=False)
            )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input + context),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            data = json.loads(content)
        except Exception:
            # harter Fallback (LLM down / Müll)
            return CallToolAction(action="call_tool", tool="historical", args={"user_input": user_input})

        try:
            if data.get("action") == "call_tool":
                return CallToolAction(**data)
            if data.get("action") == "ask_user":
                return AskUserAction(**data)
        except Exception:
            pass

        return CallToolAction(action="call_tool", tool="historical", args={"user_input": user_input})