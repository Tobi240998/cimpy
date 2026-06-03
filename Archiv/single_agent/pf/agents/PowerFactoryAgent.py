import ollama #Sprachmodell
import json #Übertragungsformat
import re #Formatierung Text 

from typing import Any, Dict, Optional


class PowerFactoryAgent:

    """
    Agent responsible for:
    - executing validated actions in PowerFactory
    """
    #Lädt die Daten (das Projekt und den Studycase)
    def __init__(self, project, studycase):
        self.project = project
        self.studycase = studycase


    

    # ------------------------------------------------------------------
    # VALIDATION PART
    # ------------------------------------------------------------------
    def validate_instruction(self, instruction: dict) -> None:
        if not isinstance(instruction, dict):
            raise RuntimeError("Instruction must be a dict")

        if instruction.get("action") != "change_load":
            raise RuntimeError("Unsupported action")

        if "delta_p_mw" not in instruction:
            raise RuntimeError("Missing field: delta_p_mw")


    # ------------------------------------------------------------------
    # READ PART
    # ------------------------------------------------------------------
    #Liest den aktuellen Zustand der Last
    def read_load_state(self, load) -> Dict[str, Any]:
        p_old = load.GetAttribute("plini")

        return {
            "loc_name": getattr(load, "loc_name", None),
            "full_name": load.GetFullName() if hasattr(load, "GetFullName") else None,
            "plini_mw": p_old,
        }


    # ------------------------------------------------------------------
    # ACTION PART
    # ------------------------------------------------------------------
    #Änderung der Last
    def apply_load_change(self, load, delta_p_mw: float) -> Dict[str, Any]:
        state_before = self.read_load_state(load)

        p_old = state_before["plini_mw"]
        p_new = p_old + delta_p_mw

        load.SetAttribute("plini", p_new)

        state_after = self.read_load_state(load)

        return {
            "action": "change_load",
            "load_name": state_after["loc_name"],
            "full_name": state_after["full_name"],
            "delta_p_mw": delta_p_mw,
            "p_old_mw": p_old,
            "p_new_mw": p_new,
            "state_before": state_before,
            "state_after": state_after,
        }


    # ------------------------------------------------------------------
    # RESULT PART
    # ------------------------------------------------------------------
    def build_execution_result(self, instruction: dict, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "ok",
            "tool": "powerfactory",
            "instruction": instruction,
            "execution": execution_data,
        }


    

    #Änderung der Last 
    def execute(self, instruction: dict, load):
        self.validate_instruction(instruction)

        execution_data = self.apply_load_change(
            load=load,
            delta_p_mw=instruction["delta_p_mw"],
        )

        print(
            f"{execution_data['load_name']}: "
            f"{execution_data['p_old_mw']} → {execution_data['p_new_mw']} MW"
        )

        return self.build_execution_result(
            instruction=instruction,
            execution_data=execution_data,
        )