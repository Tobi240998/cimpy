import ollama #Sprachmodell
import json #Übertragungsformat
import re #Formatierung Text 



class PowerFactoryAgent:

    """
    Agent responsible for:
    - executing validated actions in PowerFactory
    """
    #Lädt die Daten (das Projekt und den Studycase)
    def __init__(self, project, studycase):
        self.project = project
        self.studycase = studycase


    

    #Änderung der Last 
    def execute(self, instruction: dict, load):
        if instruction["action"] != "change_load":
            raise RuntimeError("Unsupported action")

        p_old = load.GetAttribute("plini")
        p_new = p_old + instruction["delta_p_mw"]
        load.SetAttribute("plini", p_new)

        print(f"{load.loc_name}: {p_old} → {p_new} MW")

    


