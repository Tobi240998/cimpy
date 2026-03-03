from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import LoadChangeInstruction
from langchain_llm import get_llm


class LLM_interpreterAgent: 
    """
    Agent responsible for: 
    - interpreting LLM Input 
    - resolving right load
    """
    #Lädt die Daten (das Projekt, den Studycase, den Lastkatalog und den LLM-Lastkatalog)
    def __init__(self, project, studycase):
        self.project = project
        self.studycase = studycase

        self.catalog = self._build_load_catalog()
        self.load_context = self._build_llm_load_context()

        self.llm = get_llm()
        self.parser = PydanticOutputParser(
            pydantic_object=LoadChangeInstruction
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict JSON generator.\n"
             "Choose the load name from the list:\n"
             "{load_context}\n\n"
             "{format_instructions}"
            ),
            ("user", "{user_input}")
        ])

    def interpret(self, user_input: str) -> dict:
        chain = self.prompt | self.llm | self.parser

        try:
            instruction = chain.invoke({
            "user_input": user_input,
            "load_context": self.load_context,
            "format_instructions": self.parser.get_format_instructions()
        })
            return instruction.dict()

        except Exception as e:
            return {"error": "cannot_parse", "details": str(e)}


    

    # ------------------------------------------------------------------
    # RESOLUTION PART
    # ------------------------------------------------------------------
    #Definition der richtigen Last -> Vergleich Eingabe mit Token
    def resolve(self, instruction: dict):
        load_name = instruction["load_name"].lower().replace(" ", "")
        for entry in self.catalog:
            if load_name in entry["tokens"]:
                return entry["pf_object"]
        raise ValueError(f"No matching load for '{instruction['load_name']}'")
    
    # ------------------------------------------------------------------
    # CATALOG HELPERS
    # ------------------------------------------------------------------
    #Lastenkatalog
    def _build_load_catalog(self):
        catalog = []

        for load in self.project.GetContents("*.ElmLod", 1):
            entry = {
                "pf_object": load,
                "loc_name": load.loc_name,
                "full_name": load.GetFullName(),
                "tokens": self._generate_tokens(load.loc_name)
            }
            catalog.append(entry)

        return catalog
    #Generierung der Token
    def _generate_tokens(self, name: str) -> set:
        name = name.lower()
        tokens = set()

        tokens.add(name)
        tokens.add(name.replace(" ", ""))

        name_de = name.replace("load", "last")
        tokens.add(name_de)
        tokens.add(name_de.replace(" ", ""))

        return tokens
    #Definition der Lastnamen für LLM zum besseren Verständnis
    def _build_llm_load_context(self) -> str:
        lines = []
        for entry in self.catalog:
            lines.append(
                f"- {entry['loc_name']} (aliases: {', '.join(entry['tokens'])})"
            )
        return "\n".join(lines)