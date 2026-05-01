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
        "pf.build_unified_inventory",
        "pf.interpret_switch_instruction",
        "pf.resolve_objects_from_inventory_llm",
        "pf.execute_switch_operation",
        "pf.summarize_switch_result",
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
- build an executable ordered tool plan
- use only tools from the provided tool list
- do not mix CIM and PowerFactory tools in one plan
- if the data source is ambiguous, ask for clarification
- if the request is unsupported, mark unsupported

Domain guidance:
- CIM is for CIM data, historical CIM snapshots, time-based analysis, base attributes, SV values, historical comparison.
- PowerFactory is for the active PowerFactory project, simulations, load flow, switch operations, project object queries and changes.
- If the user explicitly says PowerFactory, choose PowerFactory.
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

Available standard workflows:
{available_workflows}

Workflow descriptions:

- pf.load_catalog:
  List objects (loads, lines, transformers, etc.) from the active PowerFactory project.

- pf.change_load:
  Modify load values in the PowerFactory project.

- pf.topology_query:
  Analyze connectivity and neighbors in the PowerFactory network graph.

- pf.change_switch_state:
  Open/close switches in the PowerFactory project.

- cim.standard_listing:
  List CIM objects of a certain type from snapshot inventory.

- cim.standard_base:
  Read static/base attributes from CIM data.

- cim.standard_sv:
  Read state variables (SV values) from CIM snapshot data.

- cim.standard_comparison:
  Compare CIM values across snapshots or time.

- cim.topology_query:
  Query topology relationships in CIM data, such as:
  - direct neighbors of an object
  - connected elements
  - graph connectivity
  - paths or adjacency relationships

Planning rules:
- Tool names must match the available tool list exactly.
- A plan may only contain tools from one domain.
- Use the shortest complete executable plan.
- Mutating PowerFactory tools are allowed only when the user clearly requests a change.
- Usually end with a summary tool if one exists for the workflow.
- Do not invent tools, arguments, or object names.

Available tools:
{available_tools}

Return only structured output.

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