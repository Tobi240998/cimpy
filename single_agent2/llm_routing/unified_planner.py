from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from cimpy.single_agent2.llm_routing.langchain_llm import get_llm
from cimpy.single_agent2.llm_routing.unified_plan import UnifiedPlan, UnifiedPlanStep


STANDARD_WORKFLOWS = {
    # ---------- PowerFactory ----------
    "pf.load_catalog": [
        "pf.get_load_catalog",
        "pf.summarize_load_catalog",
    ],
    "pf.change_load": [
        "pf.interpret_instruction",
        "pf.resolve_load",
        "pf.execute_change_load",
        "pf.summarize_powerfactory_result",
    ],
    "pf.topology_query": [
        "pf.build_topology_graph",
        "pf.build_topology_inventory",
        "pf.interpret_entity_instruction",
        "pf.resolve_entity_from_inventory",
        "pf.query_topology_neighbors",
        "pf.summarize_topology_result",
    ],
    "pf.change_switch_state": [
        "pf.interpret_switch_instruction",
        "pf.build_unified_inventory",
        "pf.resolve_objects_from_inventory_llm",
        "pf.execute_switch_operation",
        "pf.summarize_switch_result",
    ],

    "pf.query_element_data": [
    "pf.build_unified_inventory",
    "pf.interpret_data_query_instruction",
    "pf.classify_data_source",
    "pf.resolve_objects_from_inventory_llm",
    "pf.list_available_object_attributes",
    "pf.select_pf_object_attributes_llm",
    "pf.read_pf_object_attributes",
    "pf.summarize_pf_object_data_result",
    ],

    "pf.list_element_attributes": [
    "pf.build_unified_inventory",
    "pf.interpret_data_query_instruction",
    "pf.classify_data_source",
    "pf.resolve_objects_from_inventory_llm",
    "pf.list_available_object_attributes",
    ],

    # ---------- CIM ----------
    "cim.standard_listing": [
        "cim.scan_snapshot_inventory",
        "cim.list_equipment_of_type",
    ],
    "cim.standard_base": [
        "cim.scan_snapshot_inventory",
        "cim.resolve_cim_object",
        "cim.read_cim_base_values",
        "cim.summarize_cim_result",
    ],
    "cim.standard_sv": [
        "cim.scan_snapshot_inventory",
        "cim.resolve_cim_object",
        "cim.load_snapshot_cache",
        "cim.query_cim",
        "cim.summarize_cim_result",
    ],
    "cim.standard_comparison": [
        "cim.scan_snapshot_inventory",
        "cim.resolve_cim_object",
        "cim.resolve_cim_comparison",
        "cim.load_snapshot_cache",
        "cim.query_cim",
        "cim.read_cim_base_values",
        "cim.compare_cim_values",
        "cim.summarize_cim_result",
    ],
    "cim.topology_query": [
    "cim.scan_snapshot_inventory",
    "cim.resolve_cim_object",
    "cim.query_cim",
    "cim.summarize_cim_result",
],
}

class UnifiedPlannerStepDecision(BaseModel):
    tool: str = Field(description="Full tool name, e.g. cim.query_cim or pf.get_load_catalog")
    reasoning: str = ""


class UnifiedPlannerDecision(BaseModel):
    domain: Literal["cim", "powerfactory", "pf", "clarification_needed", "unsupported"]
    planning_mode: Literal[
        "standard_workflow",
        "custom_plan",
        "clarification_needed",
        "unsupported",
    ]

    workflow: Optional[str] = None
    custom_steps: List[str] = Field(default_factory=list)
    safe_to_execute: bool = True
    missing_context: List[str] = Field(default_factory=list)
    clarification_question: str = ""
    reasoning: str = ""

class UnifiedCompositeSubrequest(BaseModel):
    user_input: str
    depends_on_previous: bool = False


class UnifiedCompositeDecision(BaseModel):
    is_composite: bool
    subrequests: List[UnifiedCompositeSubrequest] = Field(default_factory=list)
    reasoning: str = ""

class UnifiedPlanner:
    def __init__(self, registry):
        self.registry = registry
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=UnifiedPlannerDecision)

        self.composite_parser = PydanticOutputParser(
            pydantic_object=UnifiedCompositeDecision
        )

        self.composite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
        You are the composite-request analyzer of a unified single-agent energy-grid analysis system.

        Your task:
        - decide whether the user request contains one task or multiple sequential tasks
        - split only when there are clearly multiple user goals or a later part depends on the result of an earlier part
        - keep normal single workflow requests as one request
        - do not invent missing technical details
        - keep the same language as the original user input
        - if unsure, prefer is_composite=false

        Important:
        - Do not split only because the sentence is long.
        - Do not split if the request can be answered by one standard workflow.
        - Split if the request asks to first identify objects and then query values or attributes of those identified objects.
        - Split if the second part refers to previous results using words like "diese", "deren", "dazu", "die dazugehörigen", "those", "their".
        - Do not split compound attribute requests such as "obere und untere Spannungsgrenze", "min und max", "Grenzwerte", or "limits". These can be handled by one standard data-query workflow.
        - Do not split load-change requests when the second part asks for resulting load-flow values caused by the same load modification.
        - Do split switch change requests when the second part asks for resulting load-flow values caused by the same switch change. This is handled with a separate workflow.
        Examples:
        - "Last C um 20 MW vergrößern. Welche Leitungsauslastungen ergeben sich?" -> is_composite=false
        - "Erhöhe Last A um 2 MW und zeige die Spannungen danach." -> is_composite=false
        - "Reduziere Last B um 1 MW. Wie ändern sich die Blindleistungen?" -> is_composite=false
        - "Öffne Schalter 1 und zeige danach die Busspannungen." -> is_composite=true
        - "Schließe Schalter 1. Wie hoch ist danach die Auslastung von Leitung 4-5?" -> is_composite=true
        

        Return only structured output.

        {format_instructions}
        """
            ),
            ("user", "User request:\n{user_input}")
        ])

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the unified planner of a single-agent energy-grid analysis system.

You receive:
- a user request
- a list of available tools from both CIM and PowerFactory

Your task:
- choose whether the request should be handled with CIM tools or PowerFactory tools
- choose exactly one standard workflow whenever possible
- use custom_plan only if no standard workflow fits
- use internal tools only inside custom_steps when planning_mode="custom_plan"
- do not mix CIM and PowerFactory tools in one plan
- if the data source is ambiguous, ask for clarification
- if the request is unsupported, mark unsupported

Domain guidance:
- CIM is for CIM data, historical CIM snapshots, time-based analysis, base attributes, SV values, historical comparison.
- PowerFactory is for the active PowerFactory project, simulations, load flow, switch operations, project object queries and changes.
- If the user explicitly says PowerFactory or PF or simulation tool, choose PowerFactory.
- If the user explicitly says CIM, historical data, CIM data or asks about date/time-based historical values, choose CIM.
- If the user asks for technical attributes without a clear source, choose clarification_needed.

Planning modes:
- standard_workflow: choose one workflow key from the workflow list
- custom_plan: provide explicit tool sequence
- clarification_needed
- unsupported

If possible, always prefer standard_workflow.
planning_mode MUST be exactly one of:
- standard_workflow
- custom_plan
- clarification_needed
- unsupported

Do not use field names like "workflow" or "custom_steps" as planning_mode.
Do not use internal tool names as workflow values.

Available standard workflows:
{available_workflows}

Workflow descriptions:

- pf.load_catalog:
  List objects (loads, lines, transformers, etc.) from the active PowerFactory project.

- pf.change_load:
  Modify load values in the active PowerFactory project.
  If the same request also asks for resulting load-flow values after the load change,
  such as voltages, line loading, transformer loading, active power, reactive power,
  currents, or losses, still use pf.change_load.
  Examples:
  - "Erhöhe Last A um 2 MW" -> pf.change_load
  - "Last C um 20 MW vergrößern. Welche Leitungsauslastungen ergeben sich?" -> pf.change_load

- pf.topology_query:
  Analyze connectivity and neighbors in the PowerFactory network graph.

- pf.change_switch_state:
  Change the switching state of a switch in the active PowerFactory project.
  Use this workflow for explicit commands that open, close, connect, disconnect,
  switch on/off, or toggle a switch.
  Examples:
  - "Öffne Schalter 1" -> pf.change_switch_state
  - "Schließe Schalter 1" -> pf.change_switch_state

- pf.query_element_data:
  Query values, parameters, attributes, limits, nominal values,
  load-flow values, or operational states of a specific object
  in the active PowerFactory project.

- pf.list_element_attributes:
  List available attributes, fields, parameters, or data columns of a specific PowerFactory object.
  Use this workflow when the user asks which attributes exist, not when the user asks for a specific attribute value.
  Examples:
  - "Welche Attribute gibt es für Bus 1 in PowerFactory?" -> pf.list_element_attributes
  - "Welche Parameter hat Line 4-5?" -> pf.list_element_attributes

CIM workflow descriptions:

- cim.standard_listing:
  List CIM objects of a certain type from snapshot inventory.

- cim.standard_base:
  Read static base, nameplate, or technical attributes from CIM data.
  Use this workflow for values that do not describe a time-dependent operating state.
  Typical examples: Base Voltage, rated voltage, nominal voltage, nominal power,
  impedance parameters, operating mode, technical IDs, mRID, static equipment metadata.
  German examples: Nennspannung, Bemessungsspannung, Base Voltage, Nennleistung.
  Example:
  - "Was ist r von Line 02-03?" -> cim.standard_base
  - "Was ist die Base Voltage von Bus 3?" -> cim.standard_base
  Important distinction:
  If the request asks for dynamic state values such as voltage, power, P, Q, S,
  or values at a specific date/time, prefer cim.standard_sv unless the wording clearly
  refers to static/rated/base attributes.

- cim.standard_sv:
  Read dynamic state-variable values from CIM snapshot data.
  Use this workflow for operating-state values that may vary by snapshot or time.
  Typical examples: voltage magnitude, voltage angle, active power, reactive power,
  apparent power, P, Q, S, current, breaker/switch state, or similar SV values.
  Example:
  - "Was war die Spannung von Bus 3 am 2026-01-09?" -> cim.standard_sv
  - "Wie hoch war die Wirkleistung von Line 02-03?" -> cim.standard_sv
  Important distinction:
  If the request asks for loading, utilization, overload, limit checks, threshold violations,
  or comparison against base/reference values, prefer cim.standard_comparison.
  If the request asks for static rated/base attributes, use cim.standard_base.

- cim.standard_comparison:
  Compare CIM values across snapshots or compare dynamic SV values against base/reference/limit values.
  Use this workflow for loading, utilization, overload, threshold violations,
  voltage limit checks, min/max comparisons, or questions such as when the highest loading occurred.
  Examples:
  - "Was war die Auslastung von Trafo 19-20 am 2026-01-09?" -> cim.standard_comparison
  - "Wie war die Spannung im Vergleich zu den Spannungsgrenzen?" -> cim.standard_comparison
  - "Wann war die höchste Auslastung von Trafo 19-20?" -> cim.standard_comparison

- cim.topology_query:
  Query topology relationships in CIM data, such as:
  - direct neighbors of an object
  - connected elements
  - graph connectivity
  - paths or adjacency relationships

Planning rules for custom_plan steps:
- Tool names must match the available tool list exactly.
- A plan may only contain tools from one domain.
- Use the shortest complete executable plan.
- Mutating PowerFactory tools are allowed only when the user clearly requests a change.
- Usually end with a summary tool if one exists for the workflow.
- Do not invent tools, arguments, or object names.

Internal tools:
The following tools are background information only.
Use them only if planning_mode="custom_plan".
Never use internal tool names as workflow values.

{available_tools}


Return only structured output. 
Do not use tools as standard workflows.  

{format_instructions}
"""
            ),
            ("user", "User request:\n{user_input}")
        ])

    def _format_tools_for_prompt(self) -> str:
        lines = []

        for spec in self.registry.list_tool_specs():
            lines.append(
                f"- {spec.full_name}: "
                f"description={spec.description}; "
                f"domain={spec.domain}; "
                f"requires_state={spec.requires_state}; "
                f"produces_state={spec.produces_state}; "
                f"mutating={spec.mutating}; "
                f"is_summary={spec.is_summary}; "
                f"tags={spec.capability_tags}"
            )

        return "\n".join(lines)

    def build_plan(self, user_input: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm | self.parser

        try:
            decision = chain.invoke({
                "user_input": user_input,
                "available_tools": self._format_tools_for_prompt(),
                "format_instructions": self.parser.get_format_instructions(),
                "available_workflows": "\n".join(STANDARD_WORKFLOWS.keys()),
            })

            data = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()

        except OutputParserException as exc:
            print("[DEBUG UnifiedPlanner parser error]", str(exc))
            print("[DEBUG UnifiedPlanner llm_output]", getattr(exc, "llm_output", None))
            return {
                "status": "error",
                "answer": "Der UnifiedPlanner hat keine gültige strukturierte Ausgabe erzeugt.",
                "planner_error": str(exc),
            }

        except Exception as exc:
            return {
                "status": "error",
                "answer": "Der UnifiedPlanner konnte keinen Plan erzeugen.",
                "planner_error": str(exc),
            }

                # --- Clarification ---
        if data.get("planning_mode") == "clarification_needed":
            return {
                "status": "needs_clarification",
                "question": data.get("clarification_question")
                or "Soll ich die Anfrage mit CIM-Daten oder mit dem PowerFactory-Projekt beantworten?",
                "missing_context": data.get("missing_context", []),
                "planner_decision": data,
            }

        # --- Unsupported ---
        if data.get("planning_mode") == "unsupported":
            return {
                "status": "error",
                "answer": "Die Anfrage wird aktuell nicht unterstützt.",
                "planner_decision": data,
            }

        if not data.get("safe_to_execute", False):
            return {
                "status": "error",
                "answer": "Die Anfrage wurde vom UnifiedPlanner nicht als sicher ausführbar eingestuft.",
                "planner_decision": data,
            }

        # --- Tool-Sequenz bestimmen ---
        planning_mode = data.get("planning_mode")
        tool_sequence = []

        if planning_mode == "standard_workflow":
            workflow = data.get("workflow")

            if workflow not in STANDARD_WORKFLOWS:
                return {
                    "status": "error",
                    "answer": f"Unbekannter Standard-Workflow: {workflow}",
                    "planner_decision": data,
                }

            tool_sequence = STANDARD_WORKFLOWS[workflow]

        elif planning_mode == "custom_plan":
            tool_sequence = data.get("custom_steps", []) or []

            if not tool_sequence:
                return {
                    "status": "error",
                    "answer": "Der UnifiedPlanner hat einen Custom Plan ohne Steps erzeugt.",
                    "planner_decision": data,
                }

        else:
            return {
                "status": "error",
                "answer": f"Ungültiger Planning Mode: {planning_mode}",
                "planner_decision": data,
            }

        # --- Validierung + UnifiedPlanSteps bauen ---
        steps = []
        used_prefixes = set()

        for full_tool in tool_sequence:
            if not isinstance(full_tool, str) or "." not in full_tool:
                return {
                    "status": "error",
                    "answer": f"Ungültiger Toolname im Plan: {full_tool}",
                    "planner_decision": data,
                }

            prefix, tool_name = full_tool.split(".", 1)

            if prefix not in {"cim", "pf"}:
                return {
                    "status": "error",
                    "answer": f"Unbekanntes Tool-Präfix im Plan: {prefix}",
                    "planner_decision": data,
                }

            if self.registry.get_tool_spec(full_tool) is None:
                return {
                    "status": "error",
                    "answer": f"Tool existiert nicht in der UnifiedToolRegistry: {full_tool}",
                    "planner_decision": data,
                }

            used_prefixes.add(prefix)

            step_domain = "powerfactory" if prefix == "pf" else "cim"

            steps.append(
                UnifiedPlanStep(
                    domain=step_domain,
                    tool=tool_name,
                    args={},
                    description=f"{planning_mode}: {data.get('workflow') or 'custom_plan'}",
                )
            )

        if len(used_prefixes) != 1:
            return {
                "status": "error",
                "answer": "Der UnifiedPlanner hat einen gemischten CIM/PF-Plan erzeugt. Hybrid-Pläne sind deaktiviert.",
                "planner_decision": data,
            }

        inferred_prefix = next(iter(used_prefixes))
        inferred_domain = "powerfactory" if inferred_prefix == "pf" else "cim"

        declared_domain = data.get("domain")
        if declared_domain in {"pf", "powerfactory"}:
            declared_domain = "powerfactory"
        elif declared_domain == "cim":
            declared_domain = "cim"
        else:
            declared_domain = inferred_domain

        if declared_domain != inferred_domain:
            return {
                "status": "error",
                "answer": (
                    f"Domain-Konflikt im Plan: domain={declared_domain}, "
                    f"Tools gehören aber zu {inferred_domain}."
                ),
                "planner_decision": data,
            }

        return {
            "status": "ok",
            "plan": UnifiedPlan(
                domain=inferred_domain,
                user_input=user_input,
                steps=steps,
                classification=data,
                reasoning=data.get("reasoning", ""),
            ),
            "planner_decision": data,
        }
    
    def decompose_request(self, user_input: str) -> Dict[str, Any]:
        chain = self.composite_prompt | self.llm | self.composite_parser

        try:
            decision = chain.invoke({
                "user_input": user_input,
                "format_instructions": self.composite_parser.get_format_instructions(),
            })
            data = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()

        except Exception as exc:
            return {
                "status": "ok",
                "is_composite": False,
                "subrequests": [],
                "reasoning": f"Composite analysis failed, treated as single request: {exc}",
            }

        subrequests = data.get("subrequests", []) or []

        if not data.get("is_composite", False):
            return {
                "status": "ok",
                "is_composite": False,
                "subrequests": [],
                "reasoning": data.get("reasoning", ""),
            }

        if len(subrequests) < 2:
            return {
                "status": "ok",
                "is_composite": False,
                "subrequests": [],
                "reasoning": "Composite decision had fewer than two subrequests; treated as single request.",
            }

        return {
            "status": "ok",
            "is_composite": True,
            "subrequests": subrequests,
            "reasoning": data.get("reasoning", ""),
        }