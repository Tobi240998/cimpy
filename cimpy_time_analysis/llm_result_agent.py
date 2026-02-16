from langchain_core.messages import HumanMessage, SystemMessage
from langchain_llm import get_llm


class LLM_resultAgent:
    """
    Agent to interpret PowerFactory results into natural language.
    """

    def __init__(self):
        # LLM zentral beziehen
        self.llm = get_llm()

        self.system_prompt = (
            "Du bist Experte für elektrische Netze.\n"
            "Fasse die Erkenntnisse zusammen, die sich aus den results ergeben und relevant sind in Bezug auf den User-Input."
            "Beachte in deinem Feedback auch Einheiten."
            "Gib keine allgemeinen Informationen zurück, rein auf die konkrete Simulation bezogene Informationen."
        )

    def summarize(self, results, user_input):
        human_prompt = (
            f"Nutzerfrage: {user_input}\n"
            f"Auswertung: {results}\n"
            "Antworte kurz, technisch korrekt, ohne Wiederholungen."
        )

        

        chat_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.llm.invoke(chat_messages)
        return response.content
