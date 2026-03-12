from langchain_core.messages import HumanMessage, SystemMessage
from cimpy.cimpy_time_analysis.langchain_llm import get_llm


class LLM_resultAgent:
    """
    Agent to interpret CIM-/Topologie-Ergebnisse into natural language.
    """

    def __init__(self):
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist Experte für elektrische Netze.\n"
            "Formuliere eine kurze, fachlich präzise Antwort ausschließlich auf Basis der übergebenen Auswertung.\n"
            "Nenne nur Informationen, die explizit in der Auswertung enthalten sind.\n"
            "Erfinde keine Extremwerte, keine Rangfolgen, keine Spannungs- oder Leistungswerte, wenn sie nicht explizit vorhanden sind.\n"
            "Wenn eine Frage mehr verlangt als in der Auswertung enthalten ist, dann sage klar, dass dieser Teil aus der vorliegenden Auswertung nicht hervorgeht.\n"
            "Keine allgemeinen Erklärungen, keine Spekulationen, keine Lehrbuchtexte."
        )

    def _build_metric_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Formuliere eine kurze technische Antwort.\n"
            "- Nutze nur Werte, die explizit in der Auswertung stehen.\n"
            "- Nenne wesentliche Extremwerte und Zeitbezug.\n"
            "- Falls vorhanden, erwähne Auslastung in %.\n"
            "- Keine zusätzlichen Annahmen.\n"
        )

    def _build_voltage_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Formuliere eine kurze technische Antwort.\n"
            "- Nutze nur Spannungswerte, die explizit in der Auswertung stehen.\n"
            "- Nenne Minimum, Maximum und Mittelwert, wenn vorhanden.\n"
            "- Keine zusätzlichen Annahmen.\n"
        )

    def _build_topology_neighbors_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Topologie-Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Antworte nur zur direkten topologischen Nachbarschaft.\n"
            "- Nutze nur die Anzahl, Typen und Namen aus der Auswertung.\n"
            "- Nenne KEINE Spannungs-, Leistungs- oder Extremwertaussagen, falls sie nicht explizit in der Auswertung vorkommen.\n"
            "- Wenn die Nutzerfrage zusätzlich nach Spannung/Leistung fragt, sage klar, dass das aus dieser Topologie-Auswertung allein nicht hervorgeht.\n"
        )

    def _build_topology_component_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Topologie-Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Antworte nur zur zusammenhängenden Komponente.\n"
            "- Nutze nur Komponentengröße, Knotenarten, Equipment-Typverteilung und ggf. die gelisteten Equipments.\n"
            "- Nenne KEINE höchste/niedrigste Spannung, KEINE höchste/niedrigste Leistung und KEINE Rangfolge nach Messwerten,\n"
            "  außer solche Werte stehen explizit in der Auswertung.\n"
            "- Wenn die Nutzerfrage zusätzlich nach Spannungs-/Leistungsmaximum fragt, sage klar, dass das aus der vorliegenden Topologie-Auswertung nicht hervorgeht.\n"
            "- Wenn die Liste gekürzt ist, erwähne das knapp.\n"
        )

    def _build_topology_path_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Topologie-Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Antworte kurz und konkret, ob ein topologischer Pfad gefunden wurde.\n"
            "- Wenn ja, nenne nur Pfadlänge und Stationen, die explizit in der Auswertung stehen.\n"
            "- Keine zusätzlichen Annahmen.\n"
        )

    def _build_fallback_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Antworte kurz und technisch korrekt.\n"
            "- Nutze ausschließlich Informationen aus der Auswertung.\n"
            "- Wenn die Auswertung die Nutzerfrage nicht vollständig beantwortet, sage das klar.\n"
        )

    def _build_prompt(self, results, user_input):
        if not isinstance(results, dict):
            return self._build_fallback_prompt(results, user_input)

        result_type = results.get("type")

        if result_type == "topology_neighbors":
            return self._build_topology_neighbors_prompt(results, user_input)

        if result_type == "topology_component":
            return self._build_topology_component_prompt(results, user_input)

        if result_type == "topology_path":
            return self._build_topology_path_prompt(results, user_input)

        if results.get("unit") == "kV":
            return self._build_voltage_prompt(results, user_input)

        if results.get("metric") in {"P", "Q", "S"}:
            return self._build_metric_prompt(results, user_input)

        return self._build_fallback_prompt(results, user_input)

    def summarize(self, results, user_input):
        human_prompt = self._build_prompt(results, user_input)

        chat_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(chat_messages)
        return response.content