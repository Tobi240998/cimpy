import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_llm import get_llm


class LLM_routeAgent:

    def __init__(self):
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist ein Routing-Entscheider.\n\n"
            "Regeln:\n"
            "- Wenn nach vergangenen Zeitpunkten, Durchschnitt, Maximum, "
            "Datum, 'war', 'über den Tag' gefragt wird → HISTORIC.\n"
            "- Wenn Änderungen, Schalten, Last ändern, "
            "'rechne Lastfluss', 'setze', 'ändere' verlangt werden → POWERFACTORY.\n\n"
            "Antworte ausschließlich als JSON:\n"
            '{ "route": "HISTORIC|POWERFACTORY" }'
        )

    def route(self, user_input: str):

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input)
        ]

        response = self.llm.invoke(messages)
        content = response.content.strip()

        try:
            return json.loads(content)
        except Exception:
            # falls Modell Müll ausgibt
            return {"route": "HISTORIC"}