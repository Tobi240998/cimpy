import json
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from cimpy.single_agent.llm_routing.langchain_llm import get_llm
from cimpy.single_agent.llm_routing.schemas import (
    RouterAction,
    CallToolAction,
    AskUserAction,
)


class DomainClassifier:
    def __init__(self):
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist der Domain-Classifier eines Single-Agent-Systems für Energienetzanalyse.\n\n"
            "Du entscheidest ausschließlich, ob eine Nutzeranfrage mit CIM-Daten oder mit PowerFactory bearbeitet werden soll.\n"
            "Du führst KEINE Fachplanung aus und rufst KEINE Tools auf.\n\n"
            "Du MUSST ausschließlich gültiges JSON ausgeben und sonst nichts.\n\n"
            "Erlaubte Outputs:\n"
            "1) Domain-Auswahl:\n"
            '{ "action": "call_tool", "tool": "historical|powerfactory", "args": { "user_input": "..." } }\n\n'
            "2) Rückfrage:\n"
            '{ "action": "ask_user", "question": "...", "missing_fields": ["..."], '
            '"partial": { ... }, "intended_tool": "historical|powerfactory|null" }\n\n'
            "Domain-Regeln:\n"
            "- historical bedeutet CIM-Daten / historische CIM-Snapshots / zeitbezogene CIM-Analyse.\n"
            "- powerfactory bedeutet aktuelles PowerFactory-Projekt / Simulation / Lastfluss / Projektänderung.\n"
            "- Wenn die Anfrage explizit zeitbezogen oder historisch ist, wähle historical.\n"
            "- Wenn die Anfrage explizit PowerFactory, Simulation, Lastfluss, Schalter, Projektänderung oder aktuelle Projektobjekte nennt, wähle powerfactory.\n"
            "- Wenn die Anfrage explizit CIM, CIM-Daten oder historische CIM-Daten nennt, wähle historical.\n"
            "- Technische Attributfragen ohne Quellenbezug sind mehrdeutig.\n"
            "- Wenn weder CIM/historisch noch PowerFactory klar erkennbar ist, gib ask_user zurück.\n"
            "- Die Rückfrage muss klären, ob sich die Frage auf CIM-Daten oder auf das PowerFactory-Projekt bezieht.\n"
            "- Frage nach time_range nur, wenn eine historical/CIM-Anfrage ausdrücklich zeitbezogene Werte, Verläufe, Extrema oder einen Zeitraum meint, aber der Zeitraum fehlt.\n\n"
            "Klärungskontext:\n"
            "- Wenn eine laufende Klärung vorliegt, berücksichtige ursprüngliche Nutzerfrage, letzte Rückfrage und aktuelle Nutzerantwort gemeinsam.\n"
            "- Kurze Antworten wie 'CIM', 'historische Daten', 'PowerFactory' oder 'Projekt' können die Rückfrage beantworten.\n"
            "- Wenn die Rückfrage beantwortet ist, gib call_tool zurück und stelle nicht dieselbe Rückfrage erneut.\n"
        )

    def classify(
        self,
        user_input: str,
        pending: Optional[Dict[str, Any]] = None,
    ) -> RouterAction:
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
            return AskUserAction(
                action="ask_user",
                question="Soll ich die Anfrage mit den CIM-Daten oder mit dem PowerFactory-Projekt beantworten?",
                missing_fields=["data_source"],
                partial={"user_input": user_input},
                intended_tool=None,
            )

        try:
            if data.get("action") == "call_tool":
                return CallToolAction(**data)

            if data.get("action") == "ask_user":
                return AskUserAction(**data)
        except Exception:
            pass

        return AskUserAction(
            action="ask_user",
            question="Soll ich die Anfrage mit den CIM-Daten oder mit dem PowerFactory-Projekt beantworten?",
            missing_fields=["data_source"],
            partial={"user_input": user_input},
            intended_tool=None,
        )