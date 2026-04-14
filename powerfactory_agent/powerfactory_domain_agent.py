from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from cimpy.powerfactory_agent.config import DEFAULT_PROJECT_NAME
from cimpy.powerfactory_agent.langchain_llm import get_llm
from cimpy.powerfactory_agent.powerfactory_mcp_tools import (
    build_powerfactory_services,
    _get_load_catalog_from_services,
)
from cimpy.powerfactory_agent.powerfactory_tool_registry import PowerFactoryToolRegistry


class PFPlannerDecision(BaseModel):
    intent: str = Field(
        description="One of: load_catalog, change_load, topology_query, change_switch_state, query_element_data, list_element_attributes, unsupported_powerfactory_request"
    )
    confidence: str = Field(description="One of: high, medium, low")
    target_kind: str = Field(description="Main target type, e.g. load, topology_asset, switch, catalog, unknown")
    safe_to_execute: bool = Field(description="True if the request can be executed with the currently supported workflow")
    missing_context: List[str] = Field(default_factory=list)
    required_steps: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of why this plan was selected")


class PFFlexiblePlanDecision(BaseModel):
    required_steps: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of why this step sequence was selected")


class PFCompositeSubrequest(BaseModel):
    user_input: str = Field(description="A standalone subrequest in the same language as the original user input")
    depends_on_previous: bool = Field(description="True if this subrequest should be executed after the previous one")


class PFCompositeDecomposition(BaseModel):
    is_composite: bool = Field(description="True if the original request should be split into multiple subrequests")
    subrequests: List[PFCompositeSubrequest] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the decomposition decision")


class PowerFactoryDomainAgent:
    def __init__(self, project_name: str = DEFAULT_PROJECT_NAME, debug_mode: bool = True):
        self.project_name = project_name
        self.debug_mode = debug_mode
        self.llm = get_llm()
        self.registry = PowerFactoryToolRegistry()
        self.planner_parser = PydanticOutputParser(pydantic_object=PFPlannerDecision)
        self.flexible_plan_parser = PydanticOutputParser(pydantic_object=PFFlexiblePlanDecision)
        self.decomposition_parser = PydanticOutputParser(pydantic_object=PFCompositeDecomposition)
        self._last_planning_debug: Dict[str, Any] = {}

        self.planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a planning assistant for a PowerFactory domain agent.\n"
                "Supported intents:\n"
                "- load_catalog\n"
                "- change_load\n"
                "- topology_query\n"
                "- change_switch_state\n"
                "- query_element_data\n"
                "- list_element_attributes\n"
                "- unsupported_powerfactory_request\n\n"
                "You may only use the internal steps defined below.\n"
                "Use the step contracts exactly. If a step requires state, ensure that a producer step appears earlier in the plan.\n"
                "If a request clearly combines multiple supported tasks, you may return a longer linear required_steps plan.\n"
                "Do not invent tools or step names.\n\n"
                "Step contracts:\n"
                "{step_contracts}\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            ("user", "User request:\n{user_input}"),
        ])

        self.flexible_plan_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a fallback planner for a PowerFactory domain agent.\n"
                "Build the best valid ordered step sequence from the internal step contracts below.\n\n"
                "Rules:\n"
                "- Use only the defined steps.\n"
                "- Respect requires_state and produces_state.\n"
                "- Keep the plan executable in one linear pipeline.\n"
                "- Keep the plan as short as possible, but complete.\n"
                "- If read_pf_object_attributes is used, ensure all required predecessor steps exist.\n"
                "- Prefer ending with exactly one user-facing summary step per subtask.\n"
                "- If the request cannot be mapped reliably, return [unsupported_request].\n"
                "- Do not invent new tools, steps, or branches.\n\n"
                "Step contracts:\n"
                "{step_contracts}\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            (
                "user",
                "User request:\n{user_input}\n\n"
                "Current classification:\n{classification}\n"
            ),
        ])

        self.decomposition_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a decomposition assistant for a PowerFactory domain agent.\n"
                "Your job is to decide whether a PowerFactory user request should be split into multiple sequential subrequests.\n\n"
                "Rules:\n"
                "- Keep standard single-intent requests as a single request.\n"
                "- Mark is_composite=true only if the request clearly contains multiple user goals, multiple sequential actions, or a reference that requires a previous result.\n"
                "- If you split, each subrequest must be a standalone natural-language request in the same language as the original input.\n"
                "- Preserve execution order.\n"
                "- Do not invent technical details that are not stated by the user.\n"
                "- Do not split merely because a request is long; split only if there are multiple distinct tasks.\n"
                "- References such as 'dazu', 'deren', 'diese', 'die dazugehörigen', 'those', 'their', or similar may refer to the result of the previous subrequest.\n"
                "- Especially split requests where the first part identifies PowerFactory objects and the second part asks for data of those identified objects, for example topology first and element data second.\n"
                "- Example: 'Welche Nachbarn hat Last A? Wie ist die Spannung dazu?' should become two sequential subrequests: first determine the neighbors of Last A, then ask for the voltage of those neighbors.\n"
                "- Example: 'Welche Leitungen hängen an Bus 5 und wie hoch ist deren Auslastung?' should become two sequential subrequests: first identify the connected lines, then query their loading.\n"
                "- If unsure, prefer is_composite=false.\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            (
                "user",
                "Original user request:\n{user_input}\n\n"
                "Current classification:\n{classification}\n"
            ),
        ])

        self.contextual_subrequest_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You rewrite a dependent PowerFactory follow-up request into a standalone request using the provided previous context.\n"
                "Rules:\n"
                "- Keep the same language as the original follow-up request.\n"
                "- Preserve the user's intended meaning.\n"
                "- Use the previous context only when it is needed to resolve references such as 'dazu', 'deren', 'diese', 'die dazugehörigen', 'those', 'their', or similar.\n"
                "- Do not invent PowerFactory objects that are not present in the context.\n"
                "- If the follow-up request is already standalone, return it unchanged.\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            (
                "user",
                "Previous context:\n{previous_context}\n\n"
                "Current follow-up request:\n{current_subrequest}\n"
            ),
        ])

    def _get_step_specs(self) -> Dict[str, Dict[str, Any]]:
        return self.registry.get_step_contracts()

    def _render_step_contracts_for_prompt(self) -> str:
        lines: List[str] = []
        for step_name, spec in self._get_step_specs().items():
            notes = spec.get("domain_notes", [])
            notes_text = "; ".join(notes) if notes else "-"
            lines.append(
                f"- {step_name}: "
                f"description={spec['description']}; "
                f"requires_state={spec['requires_state']}; "
                f"produces_state={spec['produces_state']}; "
                f"mutating={spec['mutating']}; "
                f"is_summary={spec['is_summary']}; "
                f"notes={notes_text}"
            )
        return "\n".join(lines)

    def build_planner_chain(self):
        return self.planner_prompt | self.llm | self.planner_parser

    def build_flexible_plan_chain(self):
        return self.flexible_plan_prompt | self.llm | self.flexible_plan_parser

    def build_decomposition_chain(self):
        return self.decomposition_prompt | self.llm | self.decomposition_parser

    def _reset_planning_debug(self) -> None:
        self._last_planning_debug = {
            "debug_mode": self.debug_mode,
            "project_name": self.project_name,
            "initial_classification": None,
            "decomposition": None,
            "subrequests": [],
            "plan_source": None,
            "final_plan_steps": [],
            "standard_plan_steps": [],
            "classification_plan_steps": [],
        }

    def classify_request(self, user_input: str) -> Dict[str, Any]:
        """
        Robust classification with retry fallback for JSON parsing failures.
        """

        def _normalize_result(raw: Any, classification_mode: str) -> Dict[str, Any] | None:
            if raw is None:
                return None
            if hasattr(raw, "model_dump"):
                result = raw.model_dump()
            elif hasattr(raw, "dict"):
                result = raw.dict()
            elif isinstance(raw, dict):
                result = dict(raw)
            else:
                return None

            if not isinstance(result, dict):
                return None

            intent = result.get("intent")
            if intent not in {
                "load_catalog",
                "change_load",
                "topology_query",
                "change_switch_state",
                "query_element_data",
                "list_element_attributes",
                "unsupported_powerfactory_request",
            }:
                return None

            required_steps = result.get("required_steps", []) if isinstance(result.get("required_steps", []), list) else []
            if (
                intent == "query_element_data"
                and "list_available_object_attributes" in required_steps
                and "read_pf_object_attributes" not in required_steps
            ):
                intent = "list_element_attributes"

            return {
                "status": "ok",
                "classification_mode": classification_mode,
                "intent": intent,
                "confidence": result.get("confidence", "low"),
                "target_kind": result.get("target_kind", "unknown"),
                "safe_to_execute": bool(result.get("safe_to_execute", intent != "unsupported_powerfactory_request")),
                "missing_context": result.get("missing_context", []) if isinstance(result.get("missing_context", []), list) else [],
                "required_steps": required_steps,
                "reasoning": result.get("reasoning", ""),
            }

        primary_error = None
        try:
            chain = self.build_planner_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "step_contracts": self._render_step_contracts_for_prompt(),
                "format_instructions": self.planner_parser.get_format_instructions(),
            })
            normalized = _normalize_result(decision, "llm")
            if normalized is not None:
                return normalized
            primary_error = "Invalid or empty classification result"
        except Exception as e:
            primary_error = str(e)

        fallback_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Classify the following PowerFactory request into exactly one supported intent.\n"
                "Supported intents:\n"
                "- load_catalog\n"
                "- change_load\n"
                "- topology_query\n"
                "- change_switch_state\n"
                "- query_element_data\n"
                "- list_element_attributes\n"
                "- unsupported_powerfactory_request\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            ("user", "User request:\n{user_input}"),
        ])

        fallback_error = None
        try:
            chain = fallback_prompt | self.llm | self.planner_parser
            decision = chain.invoke({
                "user_input": user_input,
                "format_instructions": self.planner_parser.get_format_instructions(),
            })
            normalized = _normalize_result(decision, "fallback_llm_recovery")
            if normalized is not None:
                return normalized
            fallback_error = "Fallback returned invalid classification result"
        except Exception as e:
            fallback_error = str(e)

        return {
            "status": "ok",
            "classification_mode": "forced_fallback",
            "intent": "query_element_data",
            "confidence": "low",
            "target_kind": "unknown",
            "safe_to_execute": True,
            "missing_context": [],
            "required_steps": [
            "build_unified_inventory",
            "interpret_data_query_instruction",
            "classify_data_source",
            "resolve_objects_from_inventory_llm",
            "list_available_object_attributes",
            "select_pf_object_attributes_llm",
            "read_pf_object_attributes",
            "summarize_pf_object_data_result",
        ],
            "reasoning": f"Primary failed: {primary_error} | Fallback failed: {fallback_error}",
        }

    def _decompose_request(self, user_input: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        try:
            chain = self.build_decomposition_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "classification": classification,
                "format_instructions": self.decomposition_parser.get_format_instructions(),
            })
            result = decision.dict() if hasattr(decision, "dict") else dict(decision)
            return {"status": "ok", **result}
        except Exception as e:
            return {
                "status": "error",
                "is_composite": False,
                "subrequests": [],
                "reasoning": f"LLM decomposition failed: {str(e)}",
            }

    def _get_allowed_steps(self) -> Dict[str, str]:
        return {
            step_name: spec["description"]
            for step_name, spec in self._get_step_specs().items()
        }

    def _make_plan_item(
        self,
        step: str,
        user_input_override: str | None = None,
        source_subrequest: str | None = None,
    ) -> Dict[str, Any]:
        step_specs = self._get_step_specs()
        item: Dict[str, Any] = {
            "step": step,
            "description": step_specs[step]["description"],
        }
        if user_input_override is not None:
            item["user_input_override"] = user_input_override
        if source_subrequest is not None:
            item["source_subrequest"] = source_subrequest
        return item

    def _steps_to_plan(
        self,
        steps: List[str],
        user_input_override: str | None = None,
        source_subrequest: str | None = None,
    ) -> List[Dict[str, Any]]:
        step_specs = self._get_step_specs()
        return [
            {
                "step": step,
                "description": step_specs[step]["description"],
                **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
            }
            for step in steps
            if step in step_specs
        ]

    def _plan_to_steps(self, plan: List[Dict[str, Any]]) -> List[str]:
        return [item["step"] for item in plan if isinstance(item, dict) and "step" in item]

    def _get_step_dependencies(self) -> Dict[str, List[str]]:
        return {
            step_name: spec["requires_state"]
            for step_name, spec in self._get_step_specs().items()
        }

    def _get_step_outputs(self) -> Dict[str, List[str]]:
        return {
            step_name: spec["produces_state"]
            for step_name, spec in self._get_step_specs().items()
        }

    def _get_dependency_producers(self) -> Dict[str, str]:
        producers: Dict[str, str] = {}
        for step_name, spec in self._get_step_specs().items():
            for produced_state in spec["produces_state"]:
                producers[produced_state] = step_name
        return producers

    def _validate_step_sequence(self, steps: List[str]) -> bool:
        step_specs = self._get_step_specs()

        if not steps:
            return False

        if any(step not in step_specs for step in steps):
            return False

        if "unsupported_request" in steps and steps != ["unsupported_request"]:
            return False

        available_state = {"services"}

        for step in steps:
            requires_state = step_specs[step]["requires_state"]
            if any(required_key not in available_state for required_key in requires_state):
                return False

            for produced_key in step_specs[step]["produces_state"]:
                available_state.add(produced_key)

        return True

    def _normalize_candidate_steps(self, steps: List[str]) -> List[str]:
        step_specs = self._get_step_specs()
        normalized_steps: List[str] = []

        for step in steps:
            if not isinstance(step, str):
                continue
            if step not in step_specs:
                continue
            if step == "unsupported_request":
                continue
            if normalized_steps and normalized_steps[-1] == step:
                continue
            normalized_steps.append(step)

        return normalized_steps

    def _ensure_step_dependencies_recursive(
        self,
        step: str,
        ordered_steps: List[str],
        visiting: set[str],
    ) -> None:
        dependency_producers = self._get_dependency_producers()
        step_dependencies = self._get_step_dependencies()

        if step in ordered_steps or step in visiting:
            return

        visiting.add(step)

        for dependency_key in step_dependencies.get(step, []):
            producer_step = dependency_producers.get(dependency_key)
            if producer_step:
                self._ensure_step_dependencies_recursive(
                    step=producer_step,
                    ordered_steps=ordered_steps,
                    visiting=visiting,
                )

        if step not in ordered_steps:
            ordered_steps.append(step)

        visiting.remove(step)

    def _repair_step_sequence(self, steps: List[str]) -> List[str]:
        normalized_steps = self._normalize_candidate_steps(steps)
        if not normalized_steps:
            return []

        ordered_steps: List[str] = []

        for step in normalized_steps:
            self._ensure_step_dependencies_recursive(
                step=step,
                ordered_steps=ordered_steps,
                visiting=set(),
            )

        repaired_steps: List[str] = []
        for step in ordered_steps:
            if not repaired_steps or repaired_steps[-1] != step:
                repaired_steps.append(step)

        return repaired_steps

    def normalize_required_steps(
        self,
        classification: Dict[str, Any],
        user_input_override: str | None = None,
        source_subrequest: str | None = None,
    ) -> List[Dict[str, Any]]:
        allowed_steps = self._get_allowed_steps()

        intent = classification.get("intent")
        safe_to_execute = classification.get("safe_to_execute", False)

        if intent == "load_catalog":
            return [
                {
                    "step": "get_load_catalog",
                    "description": allowed_steps["get_load_catalog"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "summarize_load_catalog",
                    "description": allowed_steps["summarize_load_catalog"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
            ]

        if intent == "change_load" and safe_to_execute:
            return [
                {
                    "step": "interpret_instruction",
                    "description": allowed_steps["interpret_instruction"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "resolve_load",
                    "description": allowed_steps["resolve_load"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "execute_change_load",
                    "description": allowed_steps["execute_change_load"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "summarize_powerfactory_result",
                    "description": allowed_steps["summarize_powerfactory_result"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
            ]

        if intent == "topology_query":
            return [
                {
                    "step": "build_topology_graph",
                    "description": allowed_steps["build_topology_graph"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "build_topology_inventory",
                    "description": allowed_steps["build_topology_inventory"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "interpret_entity_instruction",
                    "description": allowed_steps["interpret_entity_instruction"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "resolve_entity_from_inventory",
                    "description": allowed_steps["resolve_entity_from_inventory"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "query_topology_neighbors",
                    "description": allowed_steps["query_topology_neighbors"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "summarize_topology_result",
                    "description": allowed_steps["summarize_topology_result"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
            ]

        if intent == "change_switch_state":
            return [
                {
                    "step": "interpret_switch_instruction",
                    "description": allowed_steps["interpret_switch_instruction"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "build_unified_inventory",
                    "description": allowed_steps["build_unified_inventory"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "resolve_objects_from_inventory_llm",
                    "description": allowed_steps["resolve_objects_from_inventory_llm"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "execute_switch_operation",
                    "description": allowed_steps["execute_switch_operation"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "summarize_switch_result",
                    "description": allowed_steps["summarize_switch_result"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
            ]

        if intent == "list_element_attributes":
            return [
                {
                    "step": "build_unified_inventory",
                    "description": allowed_steps["build_unified_inventory"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "interpret_data_query_instruction",
                    "description": allowed_steps["interpret_data_query_instruction"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "classify_data_source",
                    "description": allowed_steps["classify_data_source"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "resolve_objects_from_inventory_llm",
                    "description": allowed_steps["resolve_objects_from_inventory_llm"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "list_available_object_attributes",
                    "description": allowed_steps["list_available_object_attributes"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
            ]

        if intent == "query_element_data":
            return [
                {
                    "step": "build_unified_inventory",
                    "description": allowed_steps["build_unified_inventory"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "interpret_data_query_instruction",
                    "description": allowed_steps["interpret_data_query_instruction"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "classify_data_source",
                    "description": allowed_steps["classify_data_source"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "resolve_objects_from_inventory_llm",
                    "description": allowed_steps["resolve_objects_from_inventory_llm"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "list_available_object_attributes",
                    "description": allowed_steps["list_available_object_attributes"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "select_pf_object_attributes_llm",
                    "description": allowed_steps["select_pf_object_attributes_llm"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "read_pf_object_attributes",
                    "description": allowed_steps["read_pf_object_attributes"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
                {
                    "step": "summarize_pf_object_data_result",
                    "description": allowed_steps["summarize_pf_object_data_result"],
                    **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                    **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
                },
            ]

        return [
            {
                "step": "unsupported_request",
                "description": allowed_steps["unsupported_request"],
                **({"user_input_override": user_input_override} if user_input_override is not None else {}),
                **({"source_subrequest": source_subrequest} if source_subrequest is not None else {}),
            }
        ]

    def _get_classification_required_steps_plan(
        self,
        classification: Dict[str, Any],
        user_input_override: str | None = None,
        source_subrequest: str | None = None,
    ) -> List[Dict[str, Any]] | None:
        raw_steps = classification.get("required_steps", []) if isinstance(classification, dict) else []
        steps = [step for step in raw_steps if isinstance(step, str)]

        if not steps:
            return None

        if self._validate_step_sequence(steps):
            return self._steps_to_plan(
                steps,
                user_input_override=user_input_override,
                source_subrequest=source_subrequest,
            )

        repaired_steps = self._repair_step_sequence(steps)
        if self._validate_step_sequence(repaired_steps):
            return self._steps_to_plan(
                repaired_steps,
                user_input_override=user_input_override,
                source_subrequest=source_subrequest,
            )

        return None

    def _build_flexible_fallback_plan(
        self,
        user_input: str,
        classification: Dict[str, Any],
        source_subrequest: str | None = None,
    ) -> List[Dict[str, Any]]:
        try:
            chain = self.build_flexible_plan_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "classification": classification,
                "step_contracts": self._render_step_contracts_for_prompt(),
                "format_instructions": self.flexible_plan_parser.get_format_instructions(),
            })
            result = decision.dict() if hasattr(decision, "dict") else dict(decision)
            steps = result.get("required_steps", []) if isinstance(result, dict) else []
            steps = [step for step in steps if isinstance(step, str)]

            if self._validate_step_sequence(steps):
                return self._steps_to_plan(
                    steps,
                    user_input_override=user_input,
                    source_subrequest=source_subrequest,
                )

            repaired_steps = self._repair_step_sequence(steps)
            if self._validate_step_sequence(repaired_steps):
                return self._steps_to_plan(
                    repaired_steps,
                    user_input_override=user_input,
                    source_subrequest=source_subrequest,
                )
        except Exception:
            pass

        return self._steps_to_plan(
            ["unsupported_request"],
            user_input_override=user_input,
            source_subrequest=source_subrequest,
        )

    def _is_standard_plan(self, plan: List[Dict[str, Any]]) -> bool:
        steps = self._plan_to_steps(plan)
        return steps != ["unsupported_request"]

    def _is_prefix_plan(self, prefix_steps: List[str], candidate_steps: List[str]) -> bool:
        return len(candidate_steps) >= len(prefix_steps) and candidate_steps[:len(prefix_steps)] == prefix_steps

    def _deduplicate_composite_steps(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not plan:
            return plan

        deduplicated: List[Dict[str, Any]] = []
        for item in plan:
            if deduplicated:
                prev = deduplicated[-1]
                if (
                    prev.get("step") == item.get("step")
                    and prev.get("user_input_override") == item.get("user_input_override")
                    and prev.get("source_subrequest") == item.get("source_subrequest")
                ):
                    continue
            deduplicated.append(item)

        return deduplicated

    def _build_nonstandard_subplan(
        self,
        user_input: str,
        classification: Dict[str, Any],
        source_subrequest: str | None = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        classification_plan = self._get_classification_required_steps_plan(
            classification=classification,
            user_input_override=user_input,
            source_subrequest=source_subrequest,
        )
        if classification_plan is not None:
            return classification_plan, "classification_required_steps"

        return self._build_flexible_fallback_plan(
            user_input=user_input,
            classification=classification,
            source_subrequest=source_subrequest,
        ), "flexible_fallback"

    def _build_subplan_for_request(
        self,
        user_input: str,
        source_subrequest: str | None = None,
    ) -> Dict[str, Any]:
        classification = self.classify_request(user_input)

        standard_plan = self.normalize_required_steps(
            classification=classification,
            user_input_override=user_input,
            source_subrequest=source_subrequest,
        )
        if self._is_standard_plan(standard_plan):
            classification_plan = self._get_classification_required_steps_plan(
                classification=classification,
                user_input_override=user_input,
                source_subrequest=source_subrequest,
            )
            standard_steps = self._plan_to_steps(standard_plan)
            classification_steps = self._plan_to_steps(classification_plan) if classification_plan is not None else []

            if classification_plan is not None and len(classification_steps) > len(standard_steps) and self._is_prefix_plan(standard_steps, classification_steps):
                return {
                    "user_input": user_input,
                    "classification": classification,
                    "plan": classification_plan,
                    "plan_source": "extended_standard_from_classification_subplan",
                }

            return {
                "user_input": user_input,
                "classification": classification,
                "plan": standard_plan,
                "plan_source": "standard_subplan",
            }

        nonstandard_plan, plan_source = self._build_nonstandard_subplan(
            user_input=user_input,
            classification=classification,
            source_subrequest=source_subrequest,
        )
        return {
            "user_input": user_input,
            "classification": classification,
            "plan": nonstandard_plan,
            "plan_source": plan_source,
        }

    def _extract_subrequest_texts(self, decomposition: Dict[str, Any]) -> List[str]:
        raw_subrequests = decomposition.get("subrequests", [])
        subrequest_texts: List[str] = []

        for item in raw_subrequests:
            if not isinstance(item, dict):
                continue
            subrequest_text = item.get("user_input", "")
            if isinstance(subrequest_text, str) and subrequest_text.strip():
                subrequest_texts.append(subrequest_text.strip())

        return subrequest_texts

    def _build_composite_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        decomposition = self._decompose_request(
            user_input=user_input,
            classification=classification,
        )

        self._last_planning_debug["decomposition"] = decomposition

        if decomposition.get("status") != "ok" or not decomposition.get("is_composite", False):
            return None

        subrequest_texts = self._extract_subrequest_texts(decomposition)
        if len(subrequest_texts) < 2:
            return None

        composite_plan: List[Dict[str, Any]] = []

        for idx, subrequest in enumerate(subrequest_texts, start=1):
            source_subrequest = f"subrequest_{idx}"
            subrequest_debug = self._build_subplan_for_request(
                user_input=subrequest,
                source_subrequest=source_subrequest,
            )
            self._last_planning_debug["subrequests"].append({
                "user_input": subrequest_debug["user_input"],
                "classification": subrequest_debug["classification"],
                "plan_source": subrequest_debug["plan_source"],
                "plan": subrequest_debug["plan"],
                "plan_steps": self._plan_to_steps(subrequest_debug["plan"]),
            })

            substeps = self._plan_to_steps(subrequest_debug["plan"])
            if substeps == ["unsupported_request"]:
                return None

            composite_plan.extend(subrequest_debug["plan"])

        composite_plan = self._deduplicate_composite_steps(composite_plan)
        composite_steps = self._plan_to_steps(composite_plan)

        if self._validate_step_sequence(composite_steps):
            return composite_plan

        repaired_steps = self._repair_step_sequence(composite_steps)
        if self._validate_step_sequence(repaired_steps):
            # falls repariert werden muss, verlieren wir die feingranulare Zuordnung;
            # dann verwenden wir den Gesamtinput als Fallback
            return self._steps_to_plan(repaired_steps, user_input_override=user_input)

        return None

    def _build_extended_standard_composite_plan(
        self,
        user_input: str,
        classification: Dict[str, Any],
        standard_plan: List[Dict[str, Any]],
        classification_plan: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]] | None:
        if classification_plan is None:
            return None

        standard_steps = self._plan_to_steps(standard_plan)
        classification_steps = self._plan_to_steps(classification_plan)

        if not classification_steps:
            return None

        if not (len(classification_steps) > len(standard_steps) and self._is_prefix_plan(standard_steps, classification_steps)):
            return None

        decomposition = self._decompose_request(
            user_input=user_input,
            classification=classification,
        )
        self._last_planning_debug["decomposition"] = decomposition

        if decomposition.get("status") != "ok" or not decomposition.get("is_composite", False):
            return None

        subrequest_texts = self._extract_subrequest_texts(decomposition)
        if len(subrequest_texts) < 2:
            return None

        composite_plan: List[Dict[str, Any]] = []

        for idx, subrequest in enumerate(subrequest_texts, start=1):
            source_subrequest = f"subrequest_{idx}"
            subrequest_debug = self._build_subplan_for_request(
                user_input=subrequest,
                source_subrequest=source_subrequest,
            )
            self._last_planning_debug["subrequests"].append({
                "user_input": subrequest_debug["user_input"],
                "classification": subrequest_debug["classification"],
                "plan_source": subrequest_debug["plan_source"],
                "plan": subrequest_debug["plan"],
                "plan_steps": self._plan_to_steps(subrequest_debug["plan"]),
            })

            substeps = self._plan_to_steps(subrequest_debug["plan"])
            if substeps == ["unsupported_request"]:
                return None

            composite_plan.extend(subrequest_debug["plan"])

        composite_plan = self._deduplicate_composite_steps(composite_plan)
        composite_steps = self._plan_to_steps(composite_plan)

        if self._validate_step_sequence(composite_steps):
            return composite_plan

        repaired_steps = self._repair_step_sequence(composite_steps)
        if self._validate_step_sequence(repaired_steps):
            return self._steps_to_plan(repaired_steps, user_input_override=user_input)

        return None

    def build_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._reset_planning_debug()
        self._last_planning_debug["initial_classification"] = classification

        standard_plan = self.normalize_required_steps(classification=classification)
        standard_steps = self._plan_to_steps(standard_plan)
        self._last_planning_debug["standard_plan_steps"] = standard_steps

        classification_plan = self._get_classification_required_steps_plan(classification=classification)
        classification_steps = self._plan_to_steps(classification_plan) if classification_plan is not None else []
        self._last_planning_debug["classification_plan_steps"] = classification_steps

        if standard_steps != ["unsupported_request"]:
            extended_composite_plan = self._build_extended_standard_composite_plan(
                user_input=user_input,
                classification=classification,
                standard_plan=standard_plan,
                classification_plan=classification_plan,
            )
            if extended_composite_plan is not None:
                self._last_planning_debug["plan_source"] = "extended_standard_from_decomposition"
                self._last_planning_debug["final_plan_steps"] = self._plan_to_steps(extended_composite_plan)
                return extended_composite_plan

            if classification_plan is not None and len(classification_steps) > len(standard_steps) and self._is_prefix_plan(standard_steps, classification_steps):
                self._last_planning_debug["plan_source"] = "extended_standard_from_classification"
                self._last_planning_debug["final_plan_steps"] = classification_steps
                return classification_plan

            self._last_planning_debug["plan_source"] = "standard"
            self._last_planning_debug["final_plan_steps"] = standard_steps
            return standard_plan

        composite_plan = self._build_composite_plan(
            user_input=user_input,
            classification=classification,
        )
        if composite_plan is not None:
            self._last_planning_debug["plan_source"] = "composite_llm"
            self._last_planning_debug["final_plan_steps"] = self._plan_to_steps(composite_plan)
            return composite_plan

        if classification_plan is not None:
            self._last_planning_debug["plan_source"] = "classification_required_steps"
            self._last_planning_debug["final_plan_steps"] = classification_steps
            return classification_plan

        fallback_plan = self._build_flexible_fallback_plan(
            user_input=user_input,
            classification=classification,
        )
        self._last_planning_debug["plan_source"] = "flexible_fallback"
        self._last_planning_debug["final_plan_steps"] = self._plan_to_steps(fallback_plan)
        return fallback_plan

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return self.registry.list_tool_specs()

    def _is_summary_step(self, step: str) -> bool:
        step_spec = self._get_step_specs().get(step, {})
        return bool(step_spec.get("is_summary", False))

    def _merge_summary_results(
        self,
        plan: List[Dict[str, Any]],
        summary: Dict[str, Any] | None,
        summary_results: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if isinstance(summary, dict) and len(summary_results) <= 1:
            return summary

        valid_summaries = [item for item in summary_results if isinstance(item, dict) and item.get("status") == "ok"]

        if not valid_summaries:
            return summary

        if len(valid_summaries) == 1:
            return valid_summaries[0]

        merged_messages: List[str] = []
        seen_messages = set()
        answer_parts: List[str] = []
        seen_answers = set()

        for item in valid_summaries:
            answer = item.get("answer", "")
            if isinstance(answer, str):
                normalized_answer = answer.strip()
                if normalized_answer and normalized_answer not in seen_answers:
                    answer_parts.append(normalized_answer)
                    seen_answers.add(normalized_answer)

            for message in item.get("messages", []) if isinstance(item.get("messages", []), list) else []:
                if isinstance(message, str):
                    normalized_message = message.strip()
                    if normalized_message and normalized_message not in seen_messages:
                        merged_messages.append(normalized_message)
                        seen_messages.add(normalized_message)

        merged_answer = "\n\n".join(answer_parts)

        return {
            "status": "ok",
            "tool": "summarize_combined_result",
            "answer": merged_answer,
            "messages": merged_messages,
            "summary_count": len(valid_summaries),
            "summary_tools": [item.get("tool") for item in valid_summaries],
            "plan_steps": [item.get("step") for item in plan if isinstance(item, dict)],
            "parts": valid_summaries,
        }

    def _simplify_debug_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if isinstance(value, list):
            if len(value) > 10:
                return {
                    "type": "list",
                    "length": len(value),
                    "preview": [self._simplify_debug_value(item) for item in value[:5]],
                }
            return [self._simplify_debug_value(item) for item in value]

        if isinstance(value, dict):
            simplified: Dict[str, Any] = {}
            for key, item in value.items():
                if key in {"services", "app", "project", "study_case_obj", "project_obj", "pf"}:
                    simplified[key] = "<omitted>"
                else:
                    simplified[key] = self._simplify_debug_value(item)
            return simplified

        return repr(value)

    def _build_tool_kwargs_debug(self, tool_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        debug_kwargs: Dict[str, Any] = {}

        for key, value in tool_kwargs.items():
            if key == "services":
                if isinstance(value, dict):
                    debug_kwargs[key] = {
                        "status": value.get("status"),
                        "project_name": value.get("project_name"),
                    }
                else:
                    debug_kwargs[key] = "<services>"
            else:
                debug_kwargs[key] = self._simplify_debug_value(value)

        return debug_kwargs

    def _build_tool_kwargs(
        self,
        step: str,
        services: Dict[str, Any],
        effective_user_input: str,
        classification: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_kwargs: Dict[str, Any] = {"services": services}

        if step == "interpret_instruction":
            tool_kwargs["user_input"] = effective_user_input
        elif step == "resolve_load":
            tool_kwargs["instruction"] = state["instruction"]
        elif step == "execute_change_load":
            tool_kwargs["instruction"] = state["instruction"]
        elif step == "summarize_powerfactory_result":
            tool_kwargs["result_payload"] = state["execution"]
            tool_kwargs["user_input"] = effective_user_input
        elif step == "summarize_load_catalog":
            tool_kwargs["catalog_result"] = state["catalog_result"]
        elif step == "build_topology_graph":
            tool_kwargs["contract_cubicles"] = True
        elif step == "build_topology_inventory":
            tool_kwargs["topology_graph_result"] = state["graph_result"]
        elif step == "interpret_entity_instruction":
            tool_kwargs["user_input"] = effective_user_input
            tool_kwargs["inventory"] = state["inventory_result"]["inventory"] if isinstance(state["inventory_result"], dict) else {}
        elif step == "resolve_entity_from_inventory":
            tool_kwargs["instruction"] = state["entity_instruction"]
            tool_kwargs["inventory"] = state["inventory_result"]["inventory"] if isinstance(state["inventory_result"], dict) else {}
            tool_kwargs["topology_graph"] = state["graph_result"]["topology_graph"] if isinstance(state["graph_result"], dict) else None
            tool_kwargs["max_matches"] = 10
        elif step == "query_topology_neighbors":
            tool_kwargs["topology_graph"] = state["graph_result"]["topology_graph"] if isinstance(state["graph_result"], dict) else None
            tool_kwargs["asset_query"] = state["entity_resolution"].get("asset_query") if isinstance(state["entity_resolution"], dict) else effective_user_input
            tool_kwargs["selected_node_id"] = (state["entity_resolution"].get("selected_match", {}) or {}).get("node_id") if isinstance(state["entity_resolution"], dict) else None
            tool_kwargs["matches"] = state["entity_resolution"].get("matches", []) if isinstance(state["entity_resolution"], dict) else []
            tool_kwargs["max_matches"] = 10
        elif step == "summarize_topology_result":
            tool_kwargs["topology_result"] = state["topology_result"]
            tool_kwargs["graph_result"] = state["graph_result"]
            tool_kwargs["inventory_result"] = state["inventory_result"]
            tool_kwargs["entity_instruction"] = state["entity_instruction"]
            tool_kwargs["entity_resolution"] = state["entity_resolution"]
        elif step == "interpret_switch_instruction":
            tool_kwargs["user_input"] = effective_user_input
        elif step == "build_unified_inventory":
            if classification.get("intent") == "change_switch_state":
                tool_kwargs["allowed_types"] = ["switch"]
            else:
                tool_kwargs["allowed_types"] = ["bus", "load", "line", "transformer", "generator", "switch"]
        elif step == "resolve_objects_from_inventory_llm":
            if classification.get("intent") == "change_switch_state":
                tool_kwargs["instruction"] = state["switch_instruction"]
            else:
                tool_kwargs["instruction"] = state["data_query_instruction"]
            tool_kwargs["inventory"] = state["unified_inventory_result"]["inventory"] if isinstance(state["unified_inventory_result"], dict) else {}
        elif step == "execute_switch_operation":
            tool_kwargs["instruction"] = state["switch_instruction"]
            tool_kwargs["resolution"] = state["object_resolution"]
            tool_kwargs["run_loadflow_after"] = True
        elif step == "summarize_switch_result":
            tool_kwargs["result_payload"] = state["switch_execution"]
            tool_kwargs["user_input"] = effective_user_input
        elif step == "interpret_data_query_instruction":
            tool_kwargs["user_input"] = effective_user_input
            tool_kwargs["inventory"] = state["unified_inventory_result"]["inventory"] if isinstance(state["unified_inventory_result"], dict) else {}
        elif step == "classify_data_source":
            tool_kwargs["instruction"] = state["data_query_instruction"]
        elif step == "list_available_object_attributes":
            tool_kwargs["instruction"] = state["data_query_instruction"]
            tool_kwargs["resolution"] = state["object_resolution"]
        elif step == "select_pf_object_attributes_llm":
            tool_kwargs["instruction"] = state["data_query_instruction"]
            tool_kwargs["resolution"] = state["object_resolution"]
            tool_kwargs["attribute_listing"] = state["data_attribute_listing"]
        elif step == "read_pf_object_attributes":
            tool_kwargs["instruction"] = (
                state["data_attribute_selection"].get("instruction")
                if isinstance(state["data_attribute_selection"], dict)
                else None
            ) or state["data_query_instruction"]
            tool_kwargs["resolution"] = state["object_resolution"]
        elif step == "summarize_pf_object_data_result":
            tool_kwargs["result_payload"] = state["data_query_execution"]
            tool_kwargs["user_input"] = effective_user_input
        elif step == "unsupported_request":
            tool_kwargs["user_input"] = effective_user_input
            tool_kwargs["classification"] = classification

        return tool_kwargs

    def _store_step_result(self, step: str, result: Dict[str, Any], state: Dict[str, Any]) -> None:
        if step == "get_load_catalog":
            state["catalog_result"] = result
        elif step == "interpret_instruction":
            state["instruction"] = result["instruction"]
        elif step == "resolve_load":
            state["resolution"] = result
        elif step == "execute_change_load":
            state["execution"] = result
        elif step == "summarize_powerfactory_result":
            state["summary"] = result
        elif step == "summarize_load_catalog":
            state["summary"] = result
        elif step == "build_topology_graph":
            state["graph_result"] = result
        elif step == "build_topology_inventory":
            state["inventory_result"] = result
        elif step == "interpret_entity_instruction":
            state["entity_instruction"] = result["instruction"]
        elif step == "resolve_entity_from_inventory":
            state["entity_resolution"] = result
        elif step == "query_topology_neighbors":
            state["topology_result"] = result
        elif step == "summarize_topology_result":
            state["summary"] = result
        elif step == "interpret_switch_instruction":
            state["switch_instruction"] = result["instruction"]
        elif step == "execute_switch_operation":
            state["switch_execution"] = result
        elif step == "summarize_switch_result":
            state["switch_summary"] = result
            state["summary"] = result
        elif step == "build_unified_inventory":
            state["unified_inventory_result"] = result
        elif step == "interpret_data_query_instruction":
            state["data_query_instruction"] = result["instruction"]
        elif step == "classify_data_source":
            state["data_query_instruction"] = result["instruction"]
            state["data_source_decision"] = result
        elif step == "resolve_objects_from_inventory_llm":
            state["object_resolution"] = result
        elif step == "list_available_object_attributes":
            state["data_attribute_listing"] = result
        elif step == "select_pf_object_attributes_llm":
            state["data_attribute_selection"] = result
        elif step == "read_pf_object_attributes":
            state["data_query_execution"] = result
        elif step == "summarize_pf_object_data_result":
            state["data_query_summary"] = result
            state["summary"] = result

    def execute_plan(
        self,
        services: Dict[str, Any],
        plan: List[Dict[str, Any]],
        user_input: str,
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        debug_trace: List[Dict[str, Any]] = []

        state: Dict[str, Any] = {
            "instruction": None,
            "resolution": None,
            "execution": None,
            "summary": None,
            "summary_results": [],
            "catalog_result": None,
            "graph_result": None,
            "inventory_result": None,
            "entity_instruction": None,
            "entity_resolution": None,
            "topology_result": None,
            "switch_instruction": None,
            "switch_execution": None,
            "switch_summary": None,
            "data_query_instruction": None,
            "data_source_decision": None,
            "data_attribute_listing": None,
            "data_attribute_selection": None,
            "data_query_execution": None,
            "data_query_summary": None,
            "unified_inventory_result": None,
            "object_resolution": None,
        }

        for item in plan:
            step = item["step"]
            effective_user_input = item.get("user_input_override", user_input)

            tool_kwargs = self._build_tool_kwargs(
                step=step,
                services=services,
                effective_user_input=effective_user_input,
                classification=classification,
                state=state,
            )

            tool_spec = self.registry.get_tool_spec(step)
            result = self.registry.invoke(step, **tool_kwargs)

            debug_trace.append({
                "step": step,
                "effective_user_input": effective_user_input,
                "source_subrequest": item.get("source_subrequest"),
                "tool_spec": {
                    "name": tool_spec.name if tool_spec else step,
                    "description": tool_spec.description if tool_spec else "",
                    "capability_tags": tool_spec.capability_tags if tool_spec else [],
                    "mutating": tool_spec.mutating if tool_spec else False,
                },
                "tool_kwargs": self._build_tool_kwargs_debug(tool_kwargs),
                "result": result,
            })

            if result["status"] != "ok":
                return self.build_error_result(error_result=result, debug_trace=debug_trace)

            self._store_step_result(step=step, result=result, state=state)

            if self._is_summary_step(step):
                state["summary_results"].append(result)

        return self.build_success_result(
            services=services,
            user_input=user_input,
            classification=classification,
            plan=plan,
            instruction=state["instruction"],
            resolution=state["resolution"],
            execution=state["execution"],
            summary=state["summary"],
            summary_results=state["summary_results"],
            catalog_result=state["catalog_result"],
            graph_result=state["graph_result"],
            inventory_result=state["inventory_result"],
            entity_instruction=state["entity_instruction"],
            entity_resolution=state["entity_resolution"],
            topology_result=state["topology_result"],
            switch_instruction=state["switch_instruction"],
            switch_execution=state["switch_execution"],
            switch_summary=state["switch_summary"],
            data_query_instruction=state["data_query_instruction"],
            object_resolution=state["object_resolution"],
            data_attribute_listing=state["data_attribute_listing"],
            data_attribute_selection=state["data_attribute_selection"],
            data_query_execution=state["data_query_execution"],
            data_query_summary=state["data_query_summary"],
            debug_trace=debug_trace,
        )


    def _build_generic_summary(
        self,
        user_input: str,
        plan: List[Dict[str, Any]],
        catalog_result: Dict[str, Any] | None,
        topology_result: Dict[str, Any] | None,
        entity_resolution: Dict[str, Any] | None,
        switch_execution: Dict[str, Any] | None,
        object_resolution: Dict[str, Any] | None,
        data_attribute_listing: Dict[str, Any] | None,
        data_query_execution: Dict[str, Any] | None,
    ) -> Dict[str, Any] | None:
        plan_steps = self._plan_to_steps(plan)

        if isinstance(data_attribute_listing, dict):
            selected_match = object_resolution.get("selected_match", {}) if isinstance(object_resolution, dict) else {}
            selected_name = selected_match.get("name") or selected_match.get("loc_name") or selected_match.get("full_name") or "Objekt"
            selected_class = selected_match.get("pf_class") or selected_match.get("class_name") or selected_match.get("type") or "unbekanntes Objekt"

            attribute_options = data_attribute_listing.get("attribute_options", [])
            attribute_names: List[str] = []
            if isinstance(attribute_options, list):
                for option in attribute_options:
                    if not isinstance(option, dict):
                        continue
                    display_name = option.get("display_name") or option.get("label") or option.get("name") or option.get("handle")
                    if isinstance(display_name, str) and display_name.strip():
                        attribute_names.append(display_name.strip())

            seen_names = set()
            deduped_names: List[str] = []
            for name in attribute_names:
                if name not in seen_names:
                    seen_names.add(name)
                    deduped_names.append(name)

            if deduped_names:
                answer = (
                    f"Verfügbare Attribute für '{selected_name}' ({selected_class}): "
                    + "; ".join(deduped_names)
                )
            else:
                answer = f"Für '{selected_name}' ({selected_class}) wurden keine verfügbaren Attribute gefunden."

            return {
                "status": "ok",
                "tool": "summarize_generic_result",
                "answer": answer,
                "messages": [answer],
                "source": "data_attribute_listing",
                "plan_steps": plan_steps,
            }

        if isinstance(data_query_execution, dict):
            payload_data = data_query_execution.get("data", {})
            if isinstance(payload_data, dict) and payload_data:
                answer = str(payload_data)
            else:
                answer = data_query_execution.get("answer", "") if isinstance(data_query_execution.get("answer", ""), str) else str(data_query_execution)

            return {
                "status": "ok",
                "tool": "summarize_generic_result",
                "answer": answer,
                "messages": [answer] if answer else [],
                "source": "data_query_execution",
                "plan_steps": plan_steps,
            }

        if isinstance(topology_result, dict):
            selected_node = topology_result.get("selected_node", {})
            selected_name = selected_node.get("name") if isinstance(selected_node, dict) else None
            neighbors = topology_result.get("neighbors", [])
            neighbor_names: List[str] = []
            if isinstance(neighbors, list):
                for item in neighbors:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("full_name") or item.get("node_id")
                        if isinstance(name, str) and name.strip():
                            neighbor_names.append(name.strip())

            if selected_name and neighbor_names:
                answer = f"Direkte Nachbarn von '{selected_name}': " + ", ".join(neighbor_names)
            elif neighbor_names:
                answer = "Direkte Nachbarn: " + ", ".join(neighbor_names)
            else:
                answer = "Es wurden keine direkten Nachbarn gefunden."

            return {
                "status": "ok",
                "tool": "summarize_generic_result",
                "answer": answer,
                "messages": [answer],
                "source": "topology_result",
                "plan_steps": plan_steps,
            }

        if isinstance(catalog_result, dict):
            loads = catalog_result.get("loads", [])
            if isinstance(loads, list):
                load_names: List[str] = []
                for item in loads:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("loc_name")
                        if isinstance(name, str) and name.strip():
                            load_names.append(name.strip())
                answer = "Verfügbare Lasten: " + ", ".join(load_names) if load_names else "Es wurden keine Lasten gefunden."
                return {
                    "status": "ok",
                    "tool": "summarize_generic_result",
                    "answer": answer,
                    "messages": [answer],
                    "source": "catalog_result",
                    "plan_steps": plan_steps,
                }

        if isinstance(switch_execution, dict):
            switch_info = switch_execution.get("switch", {})
            switch_name = switch_info.get("name") if isinstance(switch_info, dict) else None
            state_before = switch_execution.get("state_before")
            state_after = switch_execution.get("state_after")
            if switch_name:
                answer = f"Schalter '{switch_name}': Zustand vorher={state_before}, nachher={state_after}."
            else:
                answer = "Die Schalteroperation wurde ausgeführt."
            return {
                "status": "ok",
                "tool": "summarize_generic_result",
                "answer": answer,
                "messages": [answer],
                "source": "switch_execution",
                "plan_steps": plan_steps,
            }

        if isinstance(entity_resolution, dict):
            selected_match = entity_resolution.get("selected_match", {})
            name = selected_match.get("name") if isinstance(selected_match, dict) else None
            if name:
                answer = f"Aufgelöstes Objekt: {name}"
                return {
                    "status": "ok",
                    "tool": "summarize_generic_result",
                    "answer": answer,
                    "messages": [answer],
                    "source": "entity_resolution",
                    "plan_steps": plan_steps,
                }

        return None

    def build_success_result(
        self,
        services: Dict[str, Any],
        user_input: str,
        classification: Dict[str, Any],
        plan: List[Dict[str, Any]],
        instruction: Dict[str, Any] | None,
        resolution: Dict[str, Any] | None,
        execution: Dict[str, Any] | None,
        summary: Dict[str, Any] | None,
        summary_results: List[Dict[str, Any]],
        catalog_result: Dict[str, Any] | None,
        graph_result: Dict[str, Any] | None,
        inventory_result: Dict[str, Any] | None,
        entity_instruction: Dict[str, Any] | None,
        entity_resolution: Dict[str, Any] | None,
        topology_result: Dict[str, Any] | None,
        switch_instruction: Dict[str, Any] | None,
        switch_execution: Dict[str, Any] | None,
        switch_summary: Dict[str, Any] | None,
        data_query_instruction: Dict[str, Any] | None,
        object_resolution: Dict[str, Any] | None,
        data_attribute_listing: Dict[str, Any] | None,
        data_attribute_selection: Dict[str, Any] | None,
        data_query_execution: Dict[str, Any] | None,
        data_query_summary: Dict[str, Any] | None,
        debug_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        final_summary = self._merge_summary_results(
            plan=plan,
            summary=summary,
            summary_results=summary_results,
        )

        answer = final_summary.get("answer", "") if isinstance(final_summary, dict) else ""
        messages = final_summary.get("messages", []) if isinstance(final_summary, dict) else []

        if not isinstance(final_summary, dict) or not str(answer).strip():
            generic_summary = self._build_generic_summary(
                user_input=user_input,
                plan=plan,
                catalog_result=catalog_result,
                topology_result=topology_result,
                entity_resolution=entity_resolution,
                switch_execution=switch_execution,
                object_resolution=object_resolution,
                data_attribute_listing=data_attribute_listing,
                data_query_execution=data_query_execution,
            )
            if isinstance(generic_summary, dict):
                final_summary = generic_summary
                answer = final_summary.get("answer", "") if isinstance(final_summary, dict) else ""
                messages = final_summary.get("messages", []) if isinstance(final_summary, dict) else []

        result = {
            "status": "ok",
            "tool": "powerfactory",
            "agent": "PowerFactoryDomainAgent",
            "project": services.get("project_name"),
            "studycase": execution.get("studycase") if isinstance(execution, dict) else (switch_execution.get("studycase") if isinstance(switch_execution, dict) else (data_query_execution.get("studycase") if isinstance(data_query_execution, dict) else None)),
            "user_input": user_input,
            "classification": classification,
            "plan": plan,
            "available_tools": self.get_available_tools(),
            "instruction": instruction,
            "resolved_load": execution.get("resolved_load") if isinstance(execution, dict) else None,
            "data": execution.get("data", {}) if isinstance(execution, dict) else (data_query_execution.get("data", {}) if isinstance(data_query_execution, dict) else {}),
            "catalog": catalog_result.get("loads", []) if isinstance(catalog_result, dict) else [],
            "messages": messages,
            "answer": answer,
            "summary": final_summary,
            "summary_parts": summary_results,
            "topology": {
                "graph_mode": graph_result.get("graph_mode") if isinstance(graph_result, dict) else None,
                "graph_summary": graph_result.get("graph_summary", {}) if isinstance(graph_result, dict) else {},
                "build_debug": graph_result.get("build_debug", {}) if isinstance(graph_result, dict) else {},
                "inventory": inventory_result.get("inventory", {}) if isinstance(inventory_result, dict) else {},
                "instruction": entity_instruction,
                "resolution": entity_resolution,
                "selected_node": topology_result.get("selected_node") if isinstance(topology_result, dict) else None,
                "neighbor_count": topology_result.get("neighbor_count") if isinstance(topology_result, dict) else 0,
                "neighbors": topology_result.get("neighbors", []) if isinstance(topology_result, dict) else [],
                "matches": topology_result.get("matches", []) if isinstance(topology_result, dict) else [],
            },
            "switch": {
                "instruction": switch_instruction,
                "resolution": object_resolution,
                "execution": switch_execution,
                "summary": switch_summary,
            },
            "data_query": {
                "inventory": inventory_result,
                "instruction": data_query_instruction,
                "resolution": object_resolution,
                "attribute_listing": data_attribute_listing,
                "attribute_selection": data_attribute_selection,
                "execution": data_query_execution,
                "summary": data_query_summary,
            },
        }

        if self.debug_mode:
            result["debug"] = {
                "planning": self._last_planning_debug,
                "selected_equipment": {
                    "resolved_load": execution.get("resolved_load") if isinstance(execution, dict) else None,
                    "entity_resolution": entity_resolution,
                    "object_resolution": object_resolution,
                    "object_resolution": object_resolution,
                },
                "trace": debug_trace,
            }

        return result

    def build_error_result(self, error_result: Dict[str, Any], debug_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = dict(error_result)
        result["agent"] = "PowerFactoryDomainAgent"
        result["available_tools"] = self.get_available_tools()

        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            error_code = result.get("error", "unknown_error")
            details = result.get("details", "")
            if isinstance(details, str) and details.strip():
                result["answer"] = f"Die Anfrage konnte nicht sauber ausgeführt werden ({error_code}). Details: {details}"
            else:
                result["answer"] = f"Die Anfrage konnte nicht sauber ausgeführt werden ({error_code})."

        result["debug"] = {
            "planning": self._last_planning_debug,
            "trace": debug_trace,
        }
        return result

    def run(self, user_input: str) -> Dict[str, Any]:
        print("\n================ DEBUG START ================")
        print(f"[INPUT] {user_input}")

        #PowerFactory starten, Projekt aktivieren etc.
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            print("[ERROR] Services konnten nicht gebaut werden")
            return services

        classification = self.classify_request(user_input)
        print("\n[CLASSIFICATION]")
        print(classification)

        plan = self.build_plan(user_input=user_input, classification=classification)

        print("\n[PLAN SOURCE]")
        print(self._last_planning_debug.get("plan_source"))

        print("\n[STANDARD PLAN STEPS]")
        print(self._last_planning_debug.get("standard_plan_steps"))

        print("\n[CLASSIFICATION PLAN STEPS]")
        print(self._last_planning_debug.get("classification_plan_steps"))

        print("\n[DECOMPOSITION]")
        print(self._last_planning_debug.get("decomposition"))

        print("\n[SUBREQUESTS]")
        for sub in self._last_planning_debug.get("subrequests", []):
            print(sub)

        print("\n[FINAL PLAN]")
        print(self._last_planning_debug.get("final_plan_steps"))

        print("\n[FINAL PLAN WITH INPUTS]")
        for item in plan:
            print({
                "step": item.get("step"),
                "user_input_override": item.get("user_input_override"),
                "source_subrequest": item.get("source_subrequest"),
            })

        result = self.execute_plan(
            services=services,
            plan=plan,
            user_input=user_input,
            classification=classification,
        )

        print("\n[FINAL RESULT ANSWER]")
        print(result.get("answer"))

        print("================ DEBUG END =================\n")

        return result

    def get_load_catalog(self) -> Dict[str, Any]:
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            return services
        return _get_load_catalog_from_services(services)