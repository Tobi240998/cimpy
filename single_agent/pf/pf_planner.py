from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from cimpy.single_agent.pf.config import DEFAULT_PROJECT_NAME
from cimpy.single_agent.pf.langchain_llm import get_llm
from cimpy.single_agent.pf.powerfactory_tool_registry import PowerFactoryToolRegistry


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


class PFPlanner:
    """PowerFactory planning-only component.

    Contains classification, decomposition and plan construction.
    It deliberately does not execute tools and does not build final PF results;
    execution is handled by UnifiedExecutor and pf_execution_utils.
    """

    def __init__(self, project_name: str = DEFAULT_PROJECT_NAME, debug_mode: bool = True, registry=None):
        self.project_name = project_name
        self.debug_mode = debug_mode
        self.llm = get_llm()
        self.registry = registry or PowerFactoryToolRegistry()
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
                "Intent guidance:\n"
                "- load_catalog: requests asking which PowerFactory objects of a supported type exist in the active project, "
                "for example loads, transformers, lines, buses, switches, or generators.\n"
                "- change_load: requests to increase, reduce, or set a load value.\n"
                "- topology_query: requests about neighbors, connectivity, topology, or attached assets.\n"
                "- change_switch_state: requests to open, close, or toggle a switch.\n"
                "- query_element_data: requests about technical values, parameters, or result values of a PowerFactory object.\n"
                "- list_element_attributes: requests asking which attributes or fields are available for an object.\n"
                "- unsupported_powerfactory_request: requests outside the currently supported workflows.\n\n"
                "Examples:\n"
                "- 'Welche Lasten gibt es in Powerfactory?' -> intent=load_catalog\n"
                "- 'Welche Trafos gibt es in Powerfactory?' -> intent=load_catalog\n"
                "- 'Welche Leitungen gibt es in Powerfactory?' -> intent=load_catalog\n"
                "- 'Welche Busse gibt es in Powerfactory?' -> intent=load_catalog\n\n"
                "Important:\n"
                "- For the standard supported intents, predefined standard plans already exist in the system.\n"
                "- You do NOT need to invent, optimize, or reconstruct a tool plan for these standard cases.\n"
                "- For standard cases, focus on selecting the correct intent, safety level, target kind, and missing context only.\n"
                "- Only use required_steps if they are obvious and directly correspond to a standard supported workflow.\n"
                "- Do not overthink step construction.\n"
                "- Do not invent tools or step names.\n"
                "- load_catalog: requests asking which objects of a supported type exist in the active PowerFactory project, for example loads, transformers, lines, buses, switches, or generators.\n"
                "- Example: 'Welche Lasten gibt es in Powerfactory?' -> intent=load_catalog \n"
                "- Example: 'Welche Trafos gibt es in Powerfactory?' -> intent=load_catalog \n"
                "- Example: 'Welche Leitungen gibt es in Powerfactory?' -> intent=load_catalog \n"
                "- If the request clearly matches a standard supported case, return the matching intent and keep required_steps minimal.\n\n"
                "Standard intent mapping guidance:\n"
                "- change_load: requests to increase/reduce/set a load value.\n"
                "- change_switch_state: requests to open/close/toggle a switch or breaker.\n"
                "- topology_query: requests about neighbors, connectivity, topology, or attached assets.\n"
                "- query_element_data: requests about technical values, parameters, or measured/result data of PowerFactory objects.\n"
                "- list_element_attributes: requests asking which attributes/fields are available for an object.\n"
                "- load_catalog: requests asking which loads are available.\n"
                "- unsupported_powerfactory_request: requests outside the currently supported workflow.\n\n"
                "Step contracts are provided only as background context. Standard plans already exist.\n\n"
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
                "- Example: 'Welche Leitungen hängen an Bus 5 und wie hoch ist deren Auslastung?' should become two sequential subrequests: first identify the connected lines, then query their loading.\n\n"
                "Important exceptions where you should keep the request as ONE single request (is_composite=false):\n"
                "- If the request is primarily a load-change request and the second part only asks for supported result metrics of that same load-change workflow, do NOT split it.\n"
                "- Supported integrated result metrics for a load-change workflow include: bus voltage, bus active power, bus reactive power, and line loading.\n"
                "- This includes phrasings such as:\n"
                "  * 'Erhöhe Last A um 2 MW. Wie verändert sich die Auslastung der Leitung 4-5?'\n"
                "  * 'Reduziere Last B um 1 MW und zeige die Spannungen danach.'\n"
                "  * 'Erhöhe Last A um 2 MW. Wie ändern sich danach die Blindleistungen?'\n"
                "- In such cases, the second part is NOT a separate user goal but a requested result view of the same underlying load-change workflow.\n"
                "- Therefore, keep is_composite=false for such requests.\n\n"
                "General decision principle:\n"
                "- If the second part can be understood as a result metric request of the first supported workflow, prefer is_composite=false.\n"
                "- If the second part requires a genuinely separate object discovery or a different operational workflow, then use is_composite=true.\n"
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
            print("[DEBUG classify_request] primary planner START")
            chain = self.build_planner_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "step_contracts": self._render_step_contracts_for_prompt(),
                "format_instructions": self.planner_parser.get_format_instructions(),
            })
            print("[DEBUG classify_request] primary planner END")
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
            print("[DEBUG classify_request] fallback planner START")
            chain = fallback_prompt | self.llm | self.planner_parser
            decision = chain.invoke({
                "user_input": user_input,
                "format_instructions": self.planner_parser.get_format_instructions(),
            })
            print("[DEBUG classify_request] fallback planner END")
            normalized = _normalize_result(decision, "fallback_llm_recovery")
            if normalized is not None:
                return normalized
            fallback_error = "Fallback returned invalid classification result"
        except Exception as e:
            fallback_error = str(e)

        return {
            "status": "ok",
            "classification_mode": "forced_fallback",
            "intent": "unsupported_powerfactory_request",
            "confidence": "low",
            "target_kind": "unknown",
            "safe_to_execute": False,
            "missing_context": ["classification_failed"],
            "required_steps": [],
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

        # ============================================================
        # 1) Immer zuerst versuchen, ob eine echte Composite-Zerlegung sinnvoll ist
        # ============================================================
        composite_plan = self._build_composite_plan(
            user_input=user_input,
            classification=classification,
        )
        if composite_plan is not None:
            self._last_planning_debug["plan_source"] = "composite_llm"
            self._last_planning_debug["final_plan_steps"] = self._plan_to_steps(composite_plan)
            return composite_plan

        # ============================================================
        # 2) Wenn der Standardplan gültig ist, wie bisher weiter
        # ============================================================
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

        # ============================================================
        # 3) Fallbacks wie bisher
        # ============================================================
        if classification_plan is not None:
            repaired_steps = self._repair_step_sequence(classification_steps)
            if self._validate_step_sequence(repaired_steps):
                repaired_plan = self._steps_to_plan(repaired_steps, user_input_override=user_input)
                self._last_planning_debug["plan_source"] = "classification_repaired"
                self._last_planning_debug["final_plan_steps"] = repaired_steps
                return repaired_plan

        flexible_plan = self._build_flexible_fallback_plan(
            user_input=user_input,
            classification=classification,
        )
        flexible_steps = self._plan_to_steps(flexible_plan)
        self._last_planning_debug["plan_source"] = "flexible_fallback"
        self._last_planning_debug["final_plan_steps"] = flexible_steps
        return flexible_plan

