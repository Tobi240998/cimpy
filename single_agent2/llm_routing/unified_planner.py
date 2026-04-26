from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from cimpy.single_agent2.llm_routing.langchain_llm import get_llm
from cimpy.single_agent2.llm_routing.unified_plan import UnifiedPlan, UnifiedPlanStep


class UnifiedPlannerStepDecision(BaseModel):
    tool: str = Field(description="Full tool name, e.g. cim.query_cim or pf.get_load_catalog")
    reasoning: str = ""


class UnifiedPlannerDecision(BaseModel):
    domain: str = Field(description="One of: cim, powerfactory, clarification_needed, unsupported")
    confidence: str = Field(description="One of: high, medium, low")
    safe_to_execute: bool
    missing_context: List[str] = Field(default_factory=list)
    clarification_question: str = ""
    steps: List[UnifiedPlannerStepDecision] = Field(default_factory=list)
    reasoning: str = ""


class UnifiedPlanner:
    def __init__(self, registry):
        self.registry = registry
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=UnifiedPlannerDecision)

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
            })

            data = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()

        except OutputParserException as exc:
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

        if data["domain"] == "clarification_needed":
            return {
                "status": "needs_clarification",
                "question": data.get("clarification_question")
                or "Soll ich die Anfrage mit CIM-Daten oder mit dem PowerFactory-Projekt beantworten?",
                "missing_context": data.get("missing_context", []),
                "planner_decision": data,
            }

        if not data.get("safe_to_execute", False):
            return {
                "status": "error",
                "answer": "Die Anfrage wurde vom UnifiedPlanner nicht als sicher ausführbar eingestuft.",
                "planner_decision": data,
            }

        steps = []
        for item in data.get("steps", []):
            full_tool = item["tool"]

            if "." not in full_tool:
                return {
                    "status": "error",
                    "answer": f"Ungültiger Toolname ohne Domain-Präfix: {full_tool}",
                    "planner_decision": data,
                }

            prefix, tool_name = full_tool.split(".", 1)
            domain = "powerfactory" if prefix == "pf" else "cim"

            steps.append(
                UnifiedPlanStep(
                    domain=domain,
                    tool=tool_name,
                    args={},
                    description=item.get("reasoning", ""),
                )
            )

        plan_domain = "powerfactory" if data["domain"] in {"pf", "powerfactory"} else "cim"

        return {
            "status": "ok",
            "plan": UnifiedPlan(
                domain=plan_domain,
                user_input=user_input,
                steps=steps,
                classification=data,
                reasoning=data.get("reasoning", ""),
            ),
            "planner_decision": data,
        }