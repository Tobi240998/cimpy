from langchain_core.messages import HumanMessage, SystemMessage
from cimpy.cimpy_time_analysis.langchain_llm import get_llm


class LLM_resultAgent:
    """
    Agent to interpret CIM/Topologie-Ergebnisse into natural language.
    """

    def __init__(self):
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist Experte für elektrische Netze.\n"
            "Fasse die Erkenntnisse zusammen, die sich aus den Ergebnissen ergeben und relevant sind in Bezug auf die Nutzerfrage.\n"
            "Beachte in deinem Feedback Einheiten und nenne nur konkrete, aus den Ergebnissen ableitbare Aussagen.\n"
            "Keine allgemeinen Erklärungen, keine Lehrbuchtexte, keine Spekulationen.\n"
            "Antworte präzise, technisch korrekt und knapp."
        )

    def _build_metric_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n"
            "Aufgabe:\n"
            "- Formuliere eine kurze technische Antwort.\n"
            "- Nenne die wesentlichen Extremwerte und den zeitlichen Bezug.\n"
            "- Falls vorhanden, erwähne Auslastung in %.\n"
            "- Keine allgemeinen Erklärungen.\n"
        )

    def _build_voltage_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n"
            "Aufgabe:\n"
            "- Formuliere eine kurze technische Antwort.\n"
            "- Nenne Spannungsminimum, Spannungsmaximum und Mittelwert.\n"
            "- Nenne den relevanten Zeitbezug.\n"
            "- Keine allgemeinen Erklärungen.\n"
        )

    def _build_topology_neighbors_prompt(self, results, user_input):
        equipment = results.get("equipment")
        equipment_type = results.get("equipment_type")
        graph_level = results.get("graph_level")
        num_neighbors = results.get("num_neighbors", 0)
        neighbor_type_counts = results.get("neighbor_type_counts", {})
        neighbors = results.get("neighbors", [])

        compact_neighbors = []
        for n in neighbors[:20]:
            compact_neighbors.append({
                "name": n.get("neighbor_equipment"),
                "type": n.get("neighbor_type"),
                "degree": n.get("neighbor_degree"),
                "shared_topology_node_ids": n.get("shared_topology_node_ids", []),
            })

        return (
            f"Nutzerfrage: {user_input}\n"
            f"Topologie-Auswertung:\n"
            f"- Equipment: {equipment}\n"
            f"- Equipment-Typ: {equipment_type}\n"
            f"- Graph-Level: {graph_level}\n"
            f"- Anzahl direkter Nachbarn: {num_neighbors}\n"
            f"- Nachbar-Typverteilung: {neighbor_type_counts}\n"
            f"- Nachbarn (gekürzt): {compact_neighbors}\n\n"
            "Aufgabe:\n"
            "- Antworte kurz und konkret zur direkten topologischen Nachbarschaft.\n"
            "- Nenne zuerst die Anzahl direkter Nachbarn.\n"
            "- Nenne dann die wichtigsten Nachbarnamen und Typen.\n"
            "- Wenn sinnvoll, erwähne, dass die Verbindung über gemeinsame ConnectivityNodes bzw. TopologicalNodes erkannt wurde.\n"
            "- Keine Vermutungen, keine allgemeinen Erklärungen.\n"
        )

    def _build_topology_component_prompt(self, results, user_input):
        equipment = results.get("equipment")
        equipment_type = results.get("equipment_type")
        graph_level = results.get("graph_level")
        component_size = results.get("component_size", 0)
        equipment_count = results.get("equipment_count", 0)
        node_kind_counts = results.get("node_kind_counts", {})
        equipment_type_counts = results.get("equipment_type_counts", {})
        listed_equipment = results.get("listed_equipment", [])
        truncated = results.get("truncated", False)

        compact_equipment = []
        for e in listed_equipment[:20]:
            compact_equipment.append({
                "name": e.get("name"),
                "type": e.get("cim_class"),
                "degree": e.get("degree"),
            })

        return (
            f"Nutzerfrage: {user_input}\n"
            f"Topologie-Auswertung:\n"
            f"- Equipment: {equipment}\n"
            f"- Equipment-Typ: {equipment_type}\n"
            f"- Graph-Level: {graph_level}\n"
            f"- Komponentengröße: {component_size}\n"
            f"- Anzahl Equipment-Knoten in der Komponente: {equipment_count}\n"
            f"- Knotenarten: {node_kind_counts}\n"
            f"- Equipment-Typverteilung: {equipment_type_counts}\n"
            f"- Equipment-Liste (gekürzt): {compact_equipment}\n"
            f"- Liste gekürzt: {truncated}\n\n"
            "Aufgabe:\n"
            "- Antworte kurz und technisch korrekt zur zusammenhängenden Komponente.\n"
            "- Nenne Größe und Zusammensetzung der Komponente.\n"
            "- Nenne einige der wichtigsten enthaltenen Equipments mit Namen und Typ.\n"
            "- Weise darauf hin, falls die Liste gekürzt ist.\n"
            "- Keine allgemeinen Erklärungen.\n"
        )

    def _build_topology_path_prompt(self, results, user_input):
        return (
            f"Nutzerfrage: {user_input}\n"
            f"Topologie-Auswertung: {results}\n\n"
            "Aufgabe:\n"
            "- Antworte kurz und konkret, ob ein topologischer Pfad gefunden wurde.\n"
            "- Wenn ja, nenne Pfadlänge und die wichtigsten Stationen.\n"
            "- Wenn nein, nenne nur den Grund aus den Ergebnissen.\n"
            "- Keine allgemeinen Erklärungen.\n"
        )

    def _build_prompt(self, results, user_input):
        if not isinstance(results, dict):
            return (
                f"Nutzerfrage: {user_input}\n"
                f"Auswertung: {results}\n"
                "Antworte kurz, technisch korrekt, ohne Wiederholungen."
            )

        result_type = results.get("type")

        if result_type == "topology_neighbors":
            return self._build_topology_neighbors_prompt(results, user_input)

        if result_type == "topology_component":
            return self._build_topology_component_prompt(results, user_input)

        if result_type == "topology_path":
            return self._build_topology_path_prompt(results, user_input)

        if result_type is not None and results.get("unit") == "kV":
            return self._build_voltage_prompt(results, user_input)

        if results.get("metric") in {"P", "Q", "S"}:
            return self._build_metric_prompt(results, user_input)

        return (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n"
            "Antworte kurz, technisch korrekt, ohne Wiederholungen."
        )

    def summarize(self, results, user_input):
        human_prompt = self._build_prompt(results, user_input)

        chat_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.llm.invoke(chat_messages)
        return response.content