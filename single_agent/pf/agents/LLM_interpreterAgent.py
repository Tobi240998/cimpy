from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from cimpy.powerfactory_agent.schemas import LoadChangeInstruction
from cimpy.powerfactory_agent.langchain_llm import get_llm


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

    # ------------------------------------------------------------------
    # INTERPRETATION PART
    # ------------------------------------------------------------------
    def build_interpretation_chain(self):
        return self.prompt | self.llm | self.parser

    def invoke_interpretation_chain(self, user_input: str):
        chain = self.build_interpretation_chain() #self.prompt: Prompt, siehe oben; self.llm: LLM, self.parser: geforderter Output 

        return chain.invoke({
            "user_input": user_input, #User-Input selbst
            "load_context": self.load_context, #Hilfskatalog (Lasten mit Token)
            "format_instructions": self.parser.get_format_instructions() #Vorgabe der erforderlichen Struktur der Antwort
        })

    def build_interpretation_result(self, instruction) -> dict:
        if hasattr(instruction, "dict"):
            return instruction.dict()
        if isinstance(instruction, dict):
            return instruction
        raise TypeError("Instruction must be a dict or pydantic model")

    def interpret(self, user_input: str) -> dict:
        try:
            #Übergabe von Informationen
            instruction = self.invoke_interpretation_chain(user_input=user_input)
            return self.build_interpretation_result(instruction)

        #Fehler wird angezeigt, falls es fehlschlägt
        except Exception as e:
            return {"error": "cannot_parse", "details": str(e)}


    # ------------------------------------------------------------------
    # RESOLUTION PART
    # ------------------------------------------------------------------
    #Definition der richtigen Last -> Vergleich Eingabe mit Token
    def normalize_load_name(self, load_name: str) -> str:
        return (load_name or "").lower().replace(" ", "")

    def resolve_candidates(self, instruction: dict):
        normalized_load_name = self.normalize_load_name(instruction["load_name"])
        matches = []

        for entry in self.catalog:
            if normalized_load_name in entry["tokens"]:
                matches.append(entry)

        return matches

    def build_resolution_result(self, entry: dict) -> dict:
        return {
            "loc_name": entry["loc_name"],
            "full_name": entry["full_name"],
            "tokens": sorted(entry["tokens"]),
            "pf_object": entry["pf_object"],
        }

    def resolve(self, instruction: dict):
        matches = self.resolve_candidates(instruction)

        if not matches:
            raise ValueError(f"No matching load for '{instruction['load_name']}'")

        return matches[0]["pf_object"]
    
    # ------------------------------------------------------------------
    # MCP / TOOL HELPERS
    # ------------------------------------------------------------------
    def get_load_catalog_metadata(self):
        results = []

        for entry in self.catalog:
            results.append({
                "loc_name": entry["loc_name"],
                "full_name": entry["full_name"],
                "tokens": sorted(entry["tokens"]),
            })

        return results

    def resolve_with_metadata(self, instruction: dict) -> dict:
        matches = self.resolve_candidates(instruction)

        if not matches:
            return {
                "status": "not_found",
                "requested_load_name": instruction.get("load_name"),
                "matches": [],
            }

        return {
            "status": "ok",
            "requested_load_name": instruction.get("load_name"),
            "matches": [self.build_resolution_result(entry) for entry in matches],
            "selected": self.build_resolution_result(matches[0]),
        }
    
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