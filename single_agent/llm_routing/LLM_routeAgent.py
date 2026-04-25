import json
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from cimpy.single_agent.llm_routing.langchain_llm import get_llm
from cimpy.single_agent.llm_routing.schemas import RouterAction, CallToolAction, AskUserAction


class LLM_routeAgent:
    def __init__(self):
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist ein Routing-Controller für zwei Tools: historical und powerfactory.\n\n"
            "Du MUSST ausschließlich gültiges JSON ausgeben und sonst nichts.\n\n"
            "Erlaubte Outputs:\n"
            "1) Tool Call:\n"
            '{ "action": "call_tool", "tool": "historical|powerfactory", "args": { ... } }\n\n'
            "2) Rückfrage (wenn Pflichtinfos fehlen oder die Quelle unklar ist):\n"
            '{ "action": "ask_user", "question": "...", "missing_fields": ["..."], "partial": { ... }, "intended_tool": "historical|powerfactory|null" }\n\n'
            "Routing-Regeln:\n"
            "- historical ist für historische CIM-Daten zuständig.\n"
            "- powerfactory ist für PowerFactory-Projektanfragen zuständig, inklusive lesender Abfragen und Änderungen.\n"
            "- Wenn die Anfrage explizit zeitbezogen/historisch ist (z.B. Datum, gestern, Verlauf, über den Tag, Maximum im Zeitraum), route zu historical.\n"
            "- Wenn die Anfrage explizit auf PowerFactory, eine Simulation, einen Lastfluss oder eine Projektänderung verweist, route zu powerfactory.\n"
            "- Fragen nach Zustandswerten, Parametern oder technischen Attributen sind ohne klaren Quellenbezug mehrdeutig.\n"
            "- Wenn eine solche Anfrage weder einen expliziten Zeitbezug noch einen expliziten Hinweis auf historical/CIM oder powerfactory/Projekt enthält, MUSST du ask_user zurückgeben.\n"
            "- In diesem Fall darfst du NICHT direkt historical oder powerfactory wählen.\n"
            "- Die Rückfrage muss klären, ob sich die Frage auf die historischen CIM-Daten oder auf das aktuelle PowerFactory-Projekt bezieht.\n"
            "- Wenn keine ausdrückliche Zeitsemantik vorliegt, ist time_range NICHT automatisch Pflicht.\n"
            "- Frage time_range nur dann ab, wenn eine historical-Anfrage ausdrücklich zeitbezogen ist, aber der Zeitraum fehlt.\n\n"
            "Wenn eine laufende Klärung vorliegt, berücksichtige die ursprüngliche Nutzerfrage, "
            "die zuletzt gestellte Rückfrage und die aktuelle Nutzerantwort gemeinsam.\n"
            "Die aktuelle Nutzerantwort kann kurz sein (z.B. 'historische CIM-Daten' oder 'PowerFactory-Projekt').\n"
            "Wenn die Rückfrage damit beantwortet ist, gib einen call_tool zurück und stelle nicht dieselbe Rückfrage erneut.\n\n"
            "Args-Minimum:\n"
            "- args muss mindestens {\"user_input\": <Nutzertext>} enthalten.\n\n"
            "Beispiele:\n"
            '- "Was war die Spannung von Bus 5 am 2026-01-09?" -> call_tool historical\n'
            '- "Rechne einen Lastfluss für Projekt X." -> call_tool powerfactory\n'
            '- "Wie hoch ist die Spannung von Bus 5?" -> ask_user, missing_fields=["data_source"]\n'
            '- "Was ist r von Line 02-03?" -> ask_user, missing_fields=["data_source"]\n'
            '- "Wie war die Auslastung über den Tag?" -> ask_user, intended_tool="historical", missing_fields=["time_range"]\n'
        )

    def route(self, user_input: str, pending: Optional[Dict[str, Any]] = None) -> RouterAction:
        messages = [SystemMessage(content=self.system_prompt)]

        if pending:
            partial = pending.get("partial", {}) or {}
            original_user_input = partial.get("user_input") or pending.get("user_input") or ""
            previous_question = pending.get("question") or ""

            if original_user_input:
                messages.append(HumanMessage(content=original_user_input))

            if previous_question:
                messages.append(AIMessage(content=previous_question))

            messages.append(HumanMessage(content=user_input))
        else:
            messages.append(HumanMessage(content=user_input))

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            data = json.loads(content)
        except Exception:
            return CallToolAction(action="call_tool", tool="historical", args={"user_input": user_input})

        try:
            if data.get("action") == "call_tool":
                return CallToolAction(**data)
            if data.get("action") == "ask_user":
                return AskUserAction(**data)
        except Exception:
            pass

        return CallToolAction(action="call_tool", tool="historical", args={"user_input": user_input})