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
            "Fasse technische Spannungsänderungen, die aufgrund der neuen Lastsituation entstehen, verständlich zusammen, "
            "hebe kritische Werte hervor, gib Warnungen und Empfehlungen, falls sinnvoll. "
            "Gib keine allgemeinen Informationen zurück, rein auf die konkrete Simulation bezogene Informationen."
        )

    def summarize(self, results, user_input):
        human_prompt = (
            "Ursprüngliche Nutzeranweisung:\n"
            f"{user_input}\n\n"
            "Simulationsergebnisse:\n"
            f"{results}"
        )

        chat_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.llm.invoke(chat_messages)
        return response.content
