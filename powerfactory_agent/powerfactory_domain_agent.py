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
        description="One of: load_catalog, change_load, topology_query, change_switch_state, query_element_data, unsupported_powerfactory_request"
    )
    confidence: str = Field(description="One of: high, medium, low")
    target_kind: str = Field(description="Main target type, e.g. load, topology_asset, switch, catalog, unknown")
    safe_to_execute: bool = Field(description="True if the request can be executed with the currently supported workflow")
    missing_context: List[str] = Field(default_factory=list)
    required_steps: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of required internal steps. Allowed values are: "
            "get_load_catalog, summarize_load_catalog, "
            "interpret_instruction, resolve_load, execute_change_load, summarize_powerfactory_result, "
            "build_topology_graph, build_topology_inventory, interpret_entity_instruction, "
            "resolve_entity_from_inventory, query_topology_neighbors, summarize_topology_result, "
            "interpret_switch_instruction, resolve_switch_from_inventory_llm, execute_switch_operation, summarize_switch_result, "
            "build_data_inventory, interpret_data_query_instruction, resolve_pf_object_from_inventory_llm, "
            "list_available_object_attributes, select_pf_object_attributes_llm, read_pf_object_attributes, summarize_pf_object_data_result, "
            "unsupported_request"
        )
    )
    reasoning: str = Field(description="Short explanation of why this plan was selected")


class PFFlexiblePlanDecision(BaseModel):
    required_steps: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of required internal steps. Allowed values are: "
            "get_load_catalog, summarize_load_catalog, "
            "interpret_instruction, resolve_load, execute_change_load, summarize_powerfactory_result, "
            "build_topology_graph, build_topology_inventory, interpret_entity_instruction, "
            "resolve_entity_from_inventory, query_topology_neighbors, summarize_topology_result, "
            "interpret_switch_instruction, resolve_switch_from_inventory_llm, execute_switch_operation, summarize_switch_result, "
            "build_data_inventory, interpret_data_query_instruction, resolve_pf_object_from_inventory_llm, "
            "list_available_object_attributes, select_pf_object_attributes_llm, read_pf_object_attributes, summarize_pf_object_data_result, "
            "unsupported_request"
        )
    )
    reasoning: str = Field(description="Short explanation of why this step sequence was selected")


class PFCompositeSubrequest(BaseModel):
    user_input: str = Field(description="A standalone subrequest in the same language as the original user input")
    depends_on_previous: bool = Field(description="True if this subrequest should be executed after the previous one")


class PFCompositeDecomposition(BaseModel):
    is_composite: bool = Field(description="True if the original request should be split into multiple subrequests")
    subrequests: List[PFCompositeSubrequest] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the decomposition decision")


class PowerFactoryDomainAgent:
    def __init__(self, project_name: str = DEFAULT_PROJECT_NAME):
        self.project_name = project_name
        self.llm = get_llm()
        self.registry = PowerFactoryToolRegistry()
        self.planner_parser = PydanticOutputParser(pydantic_object=PFPlannerDecision)
        self.flexible_plan_parser = PydanticOutputParser(pydantic_object=PFFlexiblePlanDecision)
        self.decomposition_parser = PydanticOutputParser(pydantic_object=PFCompositeDecomposition)

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
                "- unsupported_powerfactory_request\n\n"
                "Allowed required_steps values:\n"
                "- get_load_catalog\n"
                "- summarize_load_catalog\n"
                "- interpret_instruction\n"
                "- resolve_load\n"
                "- execute_change_load\n"
                "- summarize_powerfactory_result\n"
                "- build_topology_graph\n"
                "- build_topology_inventory\n"
                "- interpret_entity_instruction\n"
                "- resolve_entity_from_inventory\n"
                "- query_topology_neighbors\n"
                "- summarize_topology_result\n"
                "- interpret_switch_instruction\n"
                "- resolve_switch_from_inventory_llm\n"
                "- execute_switch_operation\n"
                "- summarize_switch_result\n"
                "- build_data_inventory\n"
                "- interpret_data_query_instruction\n"
                "- resolve_pf_object_from_inventory_llm\n"
                "- list_available_object_attributes\n"
                "- select_pf_object_attributes_llm\n"
                "- read_pf_object_attributes\n"
                "- summarize_pf_object_data_result\n"
                "- unsupported_request\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            ("user", "User request:\n{user_input}"),
        ])

        self.flexible_plan_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a fallback planner for a PowerFactory domain agent.\n"
                "The standard workflows remain preferred, but if no standard workflow fits exactly, build the best valid ordered step sequence from the allowed internal steps below.\n\n"
                "Rules:\n"
                "- Use only the allowed required_steps values listed below.\n"
                "- Keep the plan as short as possible.\n"
                "- The sequence must be executable in a simple linear pipeline.\n"
                "- Respect dependencies between steps.\n"
                "- If the request is unsafe or cannot be mapped reliably, return [unsupported_request].\n"
                "- Prefer ending with exactly one user-facing summary step where possible.\n"
                "- Do not invent new tools, new steps or parallel branches.\n\n"
                "Allowed required_steps values:\n"
                "- get_load_catalog\n"
                "- summarize_load_catalog\n"
                "- interpret_instruction\n"
                "- resolve_load\n"
                "- execute_change_load\n"
                "- summarize_powerfactory_result\n"
                "- build_topology_graph\n"
                "- build_topology_inventory\n"
                "- interpret_entity_instruction\n"
                "- resolve_entity_from_inventory\n"
                "- query_topology_neighbors\n"
                "- summarize_topology_result\n"
                "- interpret_switch_instruction\n"
                "- resolve_switch_from_inventory_llm\n"
                "- execute_switch_operation\n"
                "- summarize_switch_result\n"
                "- build_data_inventory\n"
                "- interpret_data_query_instruction\n"
                "- resolve_pf_object_from_inventory_llm\n"
                "- list_available_object_attributes\n"
                "- select_pf_object_attributes_llm\n"
                "- read_pf_object_attributes\n"
                "- summarize_pf_object_data_result\n"
                "- unsupported_request\n\n"
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
                "- Mark is_composite=true only if the request clearly contains multiple user goals or multiple sequential actions.\n"
                "- If you split, each subrequest must be a standalone natural-language request in the same language as the original input.\n"
                "- Preserve execution order.\n"
                "- Do not invent details that are not stated by the user.\n"
                "- Do not split merely because a request is long; split only if there are multiple distinct tasks.\n"
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

    def build_planner_chain(self):
        return self.planner_prompt | self.llm | self.planner_parser

    def build_flexible_plan_chain(self):
        return self.flexible_plan_prompt | self.llm | self.flexible_plan_parser

    def build_decomposition_chain(self):
        return self.decomposition_prompt | self.llm | self.decomposition_parser

    def classify_request(self, user_input: str) -> Dict[str, Any]:
        try:
            chain = self.build_planner_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "format_instructions": self.planner_parser.get_format_instructions(),
            })
            result = decision.dict() if hasattr(decision, "dict") else dict(decision)
            return {"status": "ok", "classification_mode": "llm", **result}
        except Exception as e:
            return {
                "status": "ok",
                "classification_mode": "fallback",
                "intent": "unsupported_powerfactory_request",
                "confidence": "low",
                "target_kind": "unknown",
                "safe_to_execute": False,
                "missing_context": [],
                "required_steps": ["unsupported_request"],
                "reasoning": f"LLM planning failed: {str(e)}",
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
            "get_load_catalog": "Read the available load catalog from the active PowerFactory project",
            "summarize_load_catalog": "Build a concise user-facing catalog answer",
            "interpret_instruction": "Interpret user input into structured PowerFactory load instruction",
            "resolve_load": "Resolve the target load inside the active PowerFactory project",
            "execute_change_load": "Apply the requested load change and run load flow before/after",
            "summarize_powerfactory_result": "Interpret the load result data and generate final user answer",
            "build_topology_graph": "Build a PowerFactory topology graph from the active project",
            "build_topology_inventory": "Build a typed inventory from the current PowerFactory topology graph",
            "interpret_entity_instruction": "Interpret user input into a structured generic PowerFactory entity instruction",
            "resolve_entity_from_inventory": "Resolve the requested entity from the PowerFactory topology inventory",
            "query_topology_neighbors": "Query neighboring assets in the PowerFactory topology graph",
            "summarize_topology_result": "Build a concise user-facing topology answer",
            "interpret_switch_instruction": "Interpret user input into a structured switch operation instruction",
            "resolve_switch_from_inventory_llm": "Resolve the requested switch via LLM-based exact candidate selection from the switch list",
            "execute_switch_operation": "Apply the requested switch state change in PowerFactory",
            "summarize_switch_result": "Summarize the switch operation result for the user",
            "build_data_inventory": "Build a lightweight typed inventory for PowerFactory data queries",
            "interpret_data_query_instruction": "Interpret a PowerFactory data query into structured object and field intent",
            "resolve_pf_object_from_inventory_llm": "Resolve the requested PowerFactory object via LLM selection from the exact candidate list",
            "list_available_object_attributes": "List the available readable attributes for the resolved PowerFactory object",
            "select_pf_object_attributes_llm": "Select the most relevant object attributes via LLM from the exact available attribute list",
            "read_pf_object_attributes": "Read the selected PowerFactory object attributes, using raw-first then loadflow-based fallback when needed",
            "summarize_pf_object_data_result": "Summarize the PowerFactory object data query result for the user",
            "unsupported_request": "Return a controlled message for unsupported PowerFactory intent",
        }

    def _steps_to_plan(self, steps: List[str]) -> List[Dict[str, Any]]:
        allowed_steps = self._get_allowed_steps()
        return [
            {"step": step, "description": allowed_steps[step]}
            for step in steps
            if step in allowed_steps
        ]

    def _get_step_dependencies(self) -> Dict[str, List[str]]:
        return {
            "get_load_catalog": [],
            "summarize_load_catalog": ["catalog_result"],
            "interpret_instruction": [],
            "resolve_load": ["instruction"],
            "execute_change_load": ["instruction"],
            "summarize_powerfactory_result": ["execution"],
            "build_topology_graph": [],
            "build_topology_inventory": ["graph_result"],
            "interpret_entity_instruction": ["inventory_result"],
            "resolve_entity_from_inventory": ["entity_instruction", "inventory_result", "graph_result"],
            "query_topology_neighbors": ["graph_result", "entity_resolution"],
            "summarize_topology_result": ["topology_result"],
            "interpret_switch_instruction": [],
            "resolve_switch_from_inventory_llm": ["switch_instruction"],
            "execute_switch_operation": ["switch_instruction", "switch_resolution"],
            "summarize_switch_result": ["switch_execution"],
            "build_data_inventory": [],
            "interpret_data_query_instruction": ["data_inventory_result"],
            "resolve_pf_object_from_inventory_llm": ["data_query_instruction", "data_inventory_result"],
            "list_available_object_attributes": ["data_query_instruction", "data_object_resolution"],
            "select_pf_object_attributes_llm": ["data_query_instruction", "data_object_resolution", "data_attribute_listing"],
            "read_pf_object_attributes": ["data_query_instruction", "data_object_resolution"],
            "summarize_pf_object_data_result": ["data_query_execution"],
            "unsupported_request": [],
        }

    def _get_step_outputs(self) -> Dict[str, List[str]]:
        return {
            "get_load_catalog": ["catalog_result"],
            "summarize_load_catalog": ["summary"],
            "interpret_instruction": ["instruction"],
            "resolve_load": ["resolution"],
            "execute_change_load": ["execution"],
            "summarize_powerfactory_result": ["summary"],
            "build_topology_graph": ["graph_result"],
            "build_topology_inventory": ["inventory_result"],
            "interpret_entity_instruction": ["entity_instruction"],
            "resolve_entity_from_inventory": ["entity_resolution"],
            "query_topology_neighbors": ["topology_result"],
            "summarize_topology_result": ["summary"],
            "interpret_switch_instruction": ["switch_instruction"],
            "resolve_switch_from_inventory_llm": ["switch_resolution"],
            "execute_switch_operation": ["switch_execution"],
            "summarize_switch_result": ["switch_summary", "summary"],
            "build_data_inventory": ["data_inventory_result"],
            "interpret_data_query_instruction": ["data_query_instruction"],
            "resolve_pf_object_from_inventory_llm": ["data_object_resolution"],
            "list_available_object_attributes": ["data_attribute_listing"],
            "select_pf_object_attributes_llm": ["data_attribute_selection"],
            "read_pf_object_attributes": ["data_query_execution"],
            "summarize_pf_object_data_result": ["data_query_summary", "summary"],
            "unsupported_request": [],
        }

    def _get_dependency_producers(self) -> Dict[str, str]:
        return {
            "catalog_result": "get_load_catalog",
            "instruction": "interpret_instruction",
            "resolution": "resolve_load",
            "execution": "execute_change_load",
            "graph_result": "build_topology_graph",
            "inventory_result": "build_topology_inventory",
            "entity_instruction": "interpret_entity_instruction",
            "entity_resolution": "resolve_entity_from_inventory",
            "topology_result": "query_topology_neighbors",
            "switch_instruction": "interpret_switch_instruction",
            "switch_resolution": "resolve_switch_from_inventory_llm",
            "switch_execution": "execute_switch_operation",
            "data_inventory_result": "build_data_inventory",
            "data_query_instruction": "interpret_data_query_instruction",
            "data_object_resolution": "resolve_pf_object_from_inventory_llm",
            "data_attribute_listing": "list_available_object_attributes",
            "data_attribute_selection": "select_pf_object_attributes_llm",
            "data_query_execution": "read_pf_object_attributes",
        }

    def _validate_step_sequence(self, steps: List[str]) -> bool:
        allowed_steps = self._get_allowed_steps()
        step_dependencies = self._get_step_dependencies()
        step_outputs = self._get_step_outputs()

        if not steps:
            return False

        if any(step not in allowed_steps for step in steps):
            return False

        if "unsupported_request" in steps and steps != ["unsupported_request"]:
            return False

        available_state = {"services"}

        for step in steps:
            required_state = step_dependencies.get(step, [])
            if any(required_key not in available_state for required_key in required_state):
                return False

            for produced_key in step_outputs.get(step, []):
                available_state.add(produced_key)

        return True

    def _normalize_candidate_steps(self, steps: List[str]) -> List[str]:
        allowed_steps = self._get_allowed_steps()
        normalized_steps: List[str] = []

        for step in steps:
            if not isinstance(step, str):
                continue
            if step not in allowed_steps:
                continue
            if step == "unsupported_request":
                continue
            if normalized_steps and normalized_steps[-1] == step:
                continue
            normalized_steps.append(step)

        if not normalized_steps:
            return []

        return normalized_steps

    def _ensure_step_dependencies_recursive(
        self,
        step: str,
        ordered_steps: List[str],
        visiting: set[str],
    ) -> None:
        dependency_producers = self._get_dependency_producers()
        step_dependencies = self._get_step_dependencies()

        if step in ordered_steps:
            return

        if step in visiting:
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

        if not ordered_steps:
            return []

        summary_steps = [step for step in ordered_steps if self._is_summary_step(step)]
        if not summary_steps:
            if "execute_change_load" in ordered_steps:
                ordered_steps.append("summarize_powerfactory_result")
            elif "query_topology_neighbors" in ordered_steps:
                ordered_steps.append("summarize_topology_result")
            elif "execute_switch_operation" in ordered_steps:
                ordered_steps.append("summarize_switch_result")
            elif "read_pf_object_attributes" in ordered_steps:
                ordered_steps.append("summarize_pf_object_data_result")
            elif "get_load_catalog" in ordered_steps:
                ordered_steps.append("summarize_load_catalog")

        repaired_steps = []
        for step in ordered_steps:
            if not repaired_steps or repaired_steps[-1] != step:
                repaired_steps.append(step)

        return repaired_steps

    def normalize_required_steps(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        allowed_steps = self._get_allowed_steps()

        intent = classification.get("intent")
        safe_to_execute = classification.get("safe_to_execute", False)

        if intent == "load_catalog":
            return [
                {"step": "get_load_catalog", "description": allowed_steps["get_load_catalog"]},
                {"step": "summarize_load_catalog", "description": allowed_steps["summarize_load_catalog"]},
            ]

        if intent == "change_load" and safe_to_execute:
            return [
                {"step": "interpret_instruction", "description": allowed_steps["interpret_instruction"]},
                {"step": "resolve_load", "description": allowed_steps["resolve_load"]},
                {"step": "execute_change_load", "description": allowed_steps["execute_change_load"]},
                {"step": "summarize_powerfactory_result", "description": allowed_steps["summarize_powerfactory_result"]},
            ]

        if intent == "topology_query":
            return [
                {"step": "build_topology_graph", "description": allowed_steps["build_topology_graph"]},
                {"step": "build_topology_inventory", "description": allowed_steps["build_topology_inventory"]},
                {"step": "interpret_entity_instruction", "description": allowed_steps["interpret_entity_instruction"]},
                {"step": "resolve_entity_from_inventory", "description": allowed_steps["resolve_entity_from_inventory"]},
                {"step": "query_topology_neighbors", "description": allowed_steps["query_topology_neighbors"]},
                {"step": "summarize_topology_result", "description": allowed_steps["summarize_topology_result"]},
            ]

        if intent == "change_switch_state":
            return [
                {"step": "interpret_switch_instruction", "description": allowed_steps["interpret_switch_instruction"]},
                {"step": "resolve_switch_from_inventory_llm", "description": allowed_steps["resolve_switch_from_inventory_llm"]},
                {"step": "execute_switch_operation", "description": allowed_steps["execute_switch_operation"]},
                {"step": "summarize_switch_result", "description": allowed_steps["summarize_switch_result"]},
            ]

        if intent == "query_element_data":
            return [
                {"step": "build_data_inventory", "description": allowed_steps["build_data_inventory"]},
                {"step": "interpret_data_query_instruction", "description": allowed_steps["interpret_data_query_instruction"]},
                {"step": "resolve_pf_object_from_inventory_llm", "description": allowed_steps["resolve_pf_object_from_inventory_llm"]},
                {"step": "list_available_object_attributes", "description": allowed_steps["list_available_object_attributes"]},
                {"step": "select_pf_object_attributes_llm", "description": allowed_steps["select_pf_object_attributes_llm"]},
                {"step": "read_pf_object_attributes", "description": allowed_steps["read_pf_object_attributes"]},
                {"step": "summarize_pf_object_data_result", "description": allowed_steps["summarize_pf_object_data_result"]},
            ]

        return [{"step": "unsupported_request", "description": allowed_steps["unsupported_request"]}]

    def _get_classification_required_steps_plan(self, classification: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        raw_steps = classification.get("required_steps", []) if isinstance(classification, dict) else []
        steps = [step for step in raw_steps if isinstance(step, str)]

        if not steps:
            return None

        if self._validate_step_sequence(steps):
            return self._steps_to_plan(steps)

        repaired_steps = self._repair_step_sequence(steps)
        if self._validate_step_sequence(repaired_steps):
            return self._steps_to_plan(repaired_steps)

        return None

    def _build_flexible_fallback_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            chain = self.build_flexible_plan_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "classification": classification,
                "format_instructions": self.flexible_plan_parser.get_format_instructions(),
            })
            result = decision.dict() if hasattr(decision, "dict") else dict(decision)
            steps = result.get("required_steps", []) if isinstance(result, dict) else []
            steps = [step for step in steps if isinstance(step, str)]

            if self._validate_step_sequence(steps):
                return self._steps_to_plan(steps)

            repaired_steps = self._repair_step_sequence(steps)
            if self._validate_step_sequence(repaired_steps):
                return self._steps_to_plan(repaired_steps)
        except Exception:
            pass

        return self._steps_to_plan(["unsupported_request"])

    def _is_standard_plan(self, plan: List[Dict[str, Any]]) -> bool:
        steps = [item["step"] for item in plan]
        return steps != ["unsupported_request"]

    def _deduplicate_composite_steps(self, steps: List[str]) -> List[str]:
        if not steps:
            return steps

        deduplicated: List[str] = []
        for step in steps:
            if deduplicated and deduplicated[-1] == step:
                continue
            deduplicated.append(step)

        return deduplicated

    def _build_nonstandard_subplan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        classification_plan = self._get_classification_required_steps_plan(classification)
        if classification_plan is not None:
            return classification_plan

        return self._build_flexible_fallback_plan(
            user_input=user_input,
            classification=classification,
        )

    def _build_subplan_for_request(self, user_input: str) -> List[Dict[str, Any]]:
        classification = self.classify_request(user_input)

        standard_plan = self.normalize_required_steps(classification)
        if self._is_standard_plan(standard_plan):
            return standard_plan

        return self._build_nonstandard_subplan(
            user_input=user_input,
            classification=classification,
        )

    def _build_composite_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        decomposition = self._decompose_request(
            user_input=user_input,
            classification=classification,
        )

        if decomposition.get("status") != "ok":
            return None

        if not decomposition.get("is_composite", False):
            return None

        raw_subrequests = decomposition.get("subrequests", [])
        subrequests = []

        for item in raw_subrequests:
            if not isinstance(item, dict):
                continue
            subrequest_text = item.get("user_input", "")
            if isinstance(subrequest_text, str) and subrequest_text.strip():
                subrequests.append(subrequest_text.strip())

        if len(subrequests) < 2:
            return None

        composite_steps: List[str] = []

        for subrequest in subrequests:
            subplan = self._build_subplan_for_request(subrequest)
            substeps = [item["step"] for item in subplan]

            if substeps == ["unsupported_request"]:
                return None

            composite_steps.extend(substeps)

        composite_steps = self._deduplicate_composite_steps(composite_steps)

        if self._validate_step_sequence(composite_steps):
            return self._steps_to_plan(composite_steps)

        repaired_steps = self._repair_step_sequence(composite_steps)
        if self._validate_step_sequence(repaired_steps):
            return self._steps_to_plan(repaired_steps)

        return None

    def build_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        standard_plan = self.normalize_required_steps(classification)
        standard_steps = [item["step"] for item in standard_plan]

        # Standardfälle bleiben deterministisch und haben immer Vorrang
        if standard_steps != ["unsupported_request"]:
            return standard_plan

        composite_plan = self._build_composite_plan(
            user_input=user_input,
            classification=classification,
        )
        if composite_plan is not None:
            return composite_plan

        classification_plan = self._get_classification_required_steps_plan(classification)
        if classification_plan is not None:
            return classification_plan

        return self._build_flexible_fallback_plan(
            user_input=user_input,
            classification=classification,
        )

    def summarize_load_catalog_result(self, catalog_result: Dict[str, Any]) -> Dict[str, Any]:
        loads = catalog_result.get("loads", [])
        names = [entry.get("loc_name") for entry in loads if entry.get("loc_name")]
        preview = names[:10]

        if not names:
            answer = "Im aktiven PowerFactory-Projekt wurden keine Lasten gefunden."
        elif len(names) <= 10:
            answer = "Verfügbare Lasten im aktiven PowerFactory-Projekt: " + ", ".join(names)
        else:
            answer = f"Im aktiven PowerFactory-Projekt wurden {len(names)} Lasten gefunden. Beispiele: " + ", ".join(preview)

        return {
            "status": "ok",
            "tool": "summarize_load_catalog",
            "answer": answer,
            "count": len(names),
            "loads": loads,
        }

    def summarize_topology_result(
        self,
        topology_result: Dict[str, Any],
        graph_result: Dict[str, Any] | None = None,
        inventory_result: Dict[str, Any] | None = None,
        entity_instruction: Dict[str, Any] | None = None,
        entity_resolution: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not topology_result or topology_result.get("status") != "ok":
            return {
                "status": "error",
                "tool": "summarize_topology_result",
                "error": "missing_topology_result",
                "answer": "Es liegt kein gültiges Topologieergebnis zur Zusammenfassung vor.",
            }

        selected = topology_result.get("selected_node", {}) or {}
        neighbors = topology_result.get("neighbors", []) or []
        selected_name = selected.get("name") or selected.get("full_name") or "<unbekannt>"
        selected_class = selected.get("pf_class") or "<unknown>"
        selected_type = selected.get("inventory_type") or "<unknown>"
        neighbor_count = topology_result.get("neighbor_count", len(neighbors))

        if neighbor_count == 0:
            answer = f"Für das Asset '{selected_name}' ({selected_class}) wurden im aktuellen PowerFactory-Topologiegraphen keine direkten Nachbarn gefunden."
        else:
            preview_items = []
            for neighbor in neighbors[:10]:
                neighbor_name = neighbor.get("name") or neighbor.get("full_name") or "<unbekannt>"
                neighbor_class = neighbor.get("pf_class") or "<unknown>"
                preview_items.append(f"{neighbor_name} ({neighbor_class})")

            if neighbor_count <= 10:
                answer = f"Direkte Nachbarn von '{selected_name}' ({selected_class}, Typ {selected_type}) im PowerFactory-Topologiegraphen: " + ", ".join(preview_items)
            else:
                answer = f"Für '{selected_name}' ({selected_class}, Typ {selected_type}) wurden {neighbor_count} direkte Nachbarn im PowerFactory-Topologiegraphen gefunden. Beispiele: " + ", ".join(preview_items)

        return {
            "status": "ok",
            "tool": "summarize_topology_result",
            "answer": answer,
            "selected_node": selected,
            "neighbor_count": neighbor_count,
            "neighbors": neighbors,
            "graph_summary": graph_result.get("graph_summary", {}) if isinstance(graph_result, dict) else {},
            "inventory_types": inventory_result.get("inventory", {}).get("available_types", []) if isinstance(inventory_result, dict) else [],
            "instruction": entity_instruction,
            "resolution": entity_resolution,
        }

    def build_unsupported_result(self, user_input: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        missing_context = classification.get("missing_context", [])
        missing_text = ""
        if missing_context:
            missing_text = " Fehlender Kontext: " + ", ".join(missing_context) + "."

        return {
            "status": "error",
            "tool": "powerfactory",
            "agent": "PowerFactoryDomainAgent",
            "error": "unsupported_powerfactory_request",
            "answer": "Die Anfrage wurde nach PowerFactory geroutet, passt aber aktuell zu keinem unterstützten PowerFactory-Ablauf oder ist noch nicht sicher ausführbar." + missing_text,
            "user_input": user_input,
            "classification": classification,
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return self.registry.list_tool_specs()

    def _is_summary_step(self, step: str) -> bool:
        return step in {
            "summarize_load_catalog",
            "summarize_powerfactory_result",
            "summarize_topology_result",
            "summarize_switch_result",
            "summarize_pf_object_data_result",
        }

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

    def _build_tool_kwargs(
        self,
        step: str,
        services: Dict[str, Any],
        user_input: str,
        classification: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        #Festlegung der Input-Parameter je Tool. Services wird immer übergeben & zusätzlich die jeweils aufgeführte Info 
        tool_kwargs: Dict[str, Any] = {"services": services}

        if step == "interpret_instruction":
            tool_kwargs["user_input"] = user_input
        elif step == "resolve_load":
            tool_kwargs["instruction"] = state["instruction"]
        elif step == "execute_change_load":
            tool_kwargs["instruction"] = state["instruction"]
        elif step == "summarize_powerfactory_result":
            tool_kwargs["result_payload"] = state["execution"]
            tool_kwargs["user_input"] = user_input
        elif step == "summarize_load_catalog":
            tool_kwargs["catalog_result"] = state["catalog_result"]
        elif step == "build_topology_graph":
            tool_kwargs["contract_cubicles"] = True
        elif step == "build_topology_inventory":
            tool_kwargs["topology_graph_result"] = state["graph_result"]
        elif step == "interpret_entity_instruction":
            tool_kwargs["user_input"] = user_input
            tool_kwargs["inventory"] = state["inventory_result"]["inventory"] if isinstance(state["inventory_result"], dict) else {}
        elif step == "resolve_entity_from_inventory":
            tool_kwargs["instruction"] = state["entity_instruction"]
            tool_kwargs["inventory"] = state["inventory_result"]["inventory"] if isinstance(state["inventory_result"], dict) else {}
            tool_kwargs["topology_graph"] = state["graph_result"]["topology_graph"] if isinstance(state["graph_result"], dict) else None
            tool_kwargs["max_matches"] = 10
        elif step == "query_topology_neighbors":
            tool_kwargs["topology_graph"] = state["graph_result"]["topology_graph"] if isinstance(state["graph_result"], dict) else None
            tool_kwargs["asset_query"] = state["entity_resolution"].get("asset_query") if isinstance(state["entity_resolution"], dict) else user_input
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
            tool_kwargs["user_input"] = user_input
        elif step == "resolve_switch_from_inventory_llm":
            tool_kwargs["instruction"] = state["switch_instruction"]
        elif step == "execute_switch_operation":
            tool_kwargs["instruction"] = state["switch_instruction"]
            tool_kwargs["resolution"] = state["switch_resolution"]
            tool_kwargs["run_loadflow_after"] = True
        elif step == "summarize_switch_result":
            tool_kwargs["result_payload"] = state["switch_execution"]
            tool_kwargs["user_input"] = user_input
        elif step == "build_data_inventory":
            pass
        elif step == "interpret_data_query_instruction":
            tool_kwargs["user_input"] = user_input
            tool_kwargs["inventory"] = state["data_inventory_result"]["inventory"] if isinstance(state["data_inventory_result"], dict) else {}
        elif step == "resolve_pf_object_from_inventory_llm":
            tool_kwargs["instruction"] = state["data_query_instruction"]
            tool_kwargs["inventory"] = state["data_inventory_result"]["inventory"] if isinstance(state["data_inventory_result"], dict) else {}
        elif step == "list_available_object_attributes":
            tool_kwargs["instruction"] = state["data_query_instruction"]
            tool_kwargs["resolution"] = state["data_object_resolution"]
        elif step == "select_pf_object_attributes_llm":
            tool_kwargs["instruction"] = state["data_query_instruction"]
            tool_kwargs["resolution"] = state["data_object_resolution"]
            tool_kwargs["attribute_listing"] = state["data_attribute_listing"]
        elif step == "read_pf_object_attributes":
            tool_kwargs["instruction"] = (state["data_attribute_selection"].get("instruction") if isinstance(state["data_attribute_selection"], dict) else None) or state["data_query_instruction"]
            tool_kwargs["resolution"] = state["data_object_resolution"]
        elif step == "summarize_pf_object_data_result":
            tool_kwargs["result_payload"] = state["data_query_execution"]
            tool_kwargs["user_input"] = user_input
        elif step == "unsupported_request":
            tool_kwargs["user_input"] = user_input
            tool_kwargs["classification"] = classification

        return tool_kwargs

    def _store_step_result(self, step: str, result: Dict[str, Any], state: Dict[str, Any]) -> None:
        #Datenfluss bei erfolgreicher Ausführung -> Speichern der Variablen
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
        elif step == "resolve_switch_from_inventory_llm":
            state["switch_resolution"] = result
        elif step == "execute_switch_operation":
            state["switch_execution"] = result
        elif step == "summarize_switch_result":
            state["switch_summary"] = result
            state["summary"] = result
        elif step == "build_data_inventory":
            state["data_inventory_result"] = result
        elif step == "interpret_data_query_instruction":
            state["data_query_instruction"] = result["instruction"]
        elif step == "resolve_pf_object_from_inventory_llm":
            state["data_object_resolution"] = result
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
            "switch_resolution": None,
            "switch_execution": None,
            "switch_summary": None,
            "data_inventory_result": None,
            "data_query_instruction": None,
            "data_object_resolution": None,
            "data_attribute_listing": None,
            "data_attribute_selection": None,
            "data_query_execution": None,
            "data_query_summary": None,
        }

        for item in plan:
            step = item["step"]

            tool_kwargs = self._build_tool_kwargs(
                step=step,
                services=services,
                user_input=user_input,
                classification=classification,
                state=state,
            )

            tool_spec = self.registry.get_tool_spec(step) #Beschreibung des Tools (aus der Registry)
            result = self.registry.invoke(step, **tool_kwargs) #Ausführung des Tools

            #Tracking für Debugging
            debug_trace.append({
                "step": step,
                "tool_spec": {
                    "name": tool_spec.name if tool_spec else step,
                    "description": tool_spec.description if tool_spec else "",
                    "capability_tags": tool_spec.capability_tags if tool_spec else [],
                    "mutating": tool_spec.mutating if tool_spec else False,
                },
                "result": result,
            })

            #Fehler-Check -> Abbruch, wenn Fehler ausgegeben wird und Rückgabe von debug_trace
            if result["status"] != "ok":
                return self.build_error_result(error_result=result, debug_trace=debug_trace)

            self._store_step_result(step=step, result=result, state=state)

            if self._is_summary_step(step):
                state["summary_results"].append(result)

        #Sammeln der Ergebnisse
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
            switch_resolution=state["switch_resolution"],
            switch_execution=state["switch_execution"],
            switch_summary=state["switch_summary"],
            data_inventory_result=state["data_inventory_result"],
            data_query_instruction=state["data_query_instruction"],
            data_object_resolution=state["data_object_resolution"],
            data_attribute_listing=state["data_attribute_listing"],
            data_attribute_selection=state["data_attribute_selection"],
            data_query_execution=state["data_query_execution"],
            data_query_summary=state["data_query_summary"],
            debug_trace=debug_trace,
        )

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
        switch_resolution: Dict[str, Any] | None,
        switch_execution: Dict[str, Any] | None,
        switch_summary: Dict[str, Any] | None,
        data_inventory_result: Dict[str, Any] | None,
        data_query_instruction: Dict[str, Any] | None,
        data_object_resolution: Dict[str, Any] | None,
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

        return {
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
                "resolution": switch_resolution,
                "execution": switch_execution,
                "summary": switch_summary,
            },
            "data_query": {
                "inventory": data_inventory_result,
                "instruction": data_query_instruction,
                "resolution": data_object_resolution,
                "attribute_listing": data_attribute_listing,
                "attribute_selection": data_attribute_selection,
                "execution": data_query_execution,
                "summary": data_query_summary,
            },
            "debug": {
                "resolution": resolution,
                "execution": execution,
                "summary": final_summary,
                "summary_parts": summary_results,
                "graph_result": graph_result,
                "inventory_result": inventory_result,
                "entity_instruction": entity_instruction,
                "entity_resolution": entity_resolution,
                "topology_result": topology_result,
                "switch_instruction": switch_instruction,
                "switch_resolution": switch_resolution,
                "switch_execution": switch_execution,
                "switch_summary": switch_summary,
                "data_inventory_result": data_inventory_result,
                "data_query_instruction": data_query_instruction,
                "data_object_resolution": data_object_resolution,
                "data_attribute_listing": data_attribute_listing,
                "data_attribute_selection": data_attribute_selection,
                "data_query_execution": data_query_execution,
                "data_query_summary": data_query_summary,
                "trace": debug_trace,
            },
        }

    def build_error_result(self, error_result: Dict[str, Any], debug_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = dict(error_result)
        result["agent"] = "PowerFactoryDomainAgent"
        result["available_tools"] = self.get_available_tools()
        result["debug"] = {"trace": debug_trace}
        return result

    def run(self, user_input: str) -> Dict[str, Any]:
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            return services

        classification = self.classify_request(user_input)
        plan = self.build_plan(user_input=user_input, classification=classification)

        return self.execute_plan(
            services=services,
            plan=plan,
            user_input=user_input,
            classification=classification,
        )

    def get_load_catalog(self) -> Dict[str, Any]:
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            return services
        return _get_load_catalog_from_services(services)