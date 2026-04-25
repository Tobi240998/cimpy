import json

from langchain_core.prompts import ChatPromptTemplate
from cimpy.powerfactory_agent.langchain_llm import get_llm


class LLM_resultAgent:
    """
    Agent responsible for:
    - summarizing PowerFactory execution results in natural language
    """

    #Lädt das LLM
    def __init__(self):
        self.llm = get_llm()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You summarize PowerFactory execution results for a user.\n"
                "Be concise, factual and clear.\n"
                "If numeric values are available, include them.\n"
                "Do not invent results that are not present in the input."
            ),
            (
                "user",
                "User request:\n"
                "{user_input}\n\n"
                "Summarize the following execution result for the end user:\n\n"
                "{result_text}"
            )
        ])

    # ------------------------------------------------------------------
    # SERIALIZATION PART
    # ------------------------------------------------------------------
    def serialize_result(self, result) -> str:
        try:
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(result)

    # ------------------------------------------------------------------
    # PROMPT PART
    # ------------------------------------------------------------------
    def build_summary_chain(self):
        return self.prompt | self.llm

    def invoke_summary_chain(self, result_text: str, user_input: str = ""):
        chain = self.build_summary_chain()
        return chain.invoke({
            "result_text": result_text,
            "user_input": user_input,
        })

    # ------------------------------------------------------------------
    # RESULT PART
    # ------------------------------------------------------------------
    def extract_summary_text(self, llm_response) -> str:
        if hasattr(llm_response, "content"):
            return llm_response.content
        return str(llm_response)

    def build_summary_result(self, raw_result, summary_text: str, user_input: str = "") -> dict:
        return {
            "status": "ok",
            "tool": "powerfactory_result_summary",
            "input_result": raw_result,
            "user_input": user_input,
            "summary": summary_text,
        }

    # ------------------------------------------------------------------
    # MCP / TOOL HELPERS
    # ------------------------------------------------------------------
    def summarize_with_metadata(self, result, user_input: str = "") -> dict:
        try:
            result_text = self.serialize_result(result)
            llm_response = self.invoke_summary_chain(
                result_text=result_text,
                user_input=user_input,
            )
            summary_text = self.extract_summary_text(llm_response)

            return self.build_summary_result(
                raw_result=result,
                summary_text=summary_text,
                user_input=user_input,
            )
        except Exception as e:
            return {
                "status": "error",
                "tool": "powerfactory_result_summary",
                "error": "summary_failed",
                "details": str(e),
                "input_result": result,
                "user_input": user_input,
            }

    #Zusammenfassung des Ergebnisses
    def summarize(self, result, user_input: str = "") -> str:
        summary_result = self.summarize_with_metadata(
            result=result,
            user_input=user_input,
        )

        if summary_result["status"] == "ok":
            return summary_result["summary"]

        return f"Fehler bei der Zusammenfassung: {summary_result['details']}"