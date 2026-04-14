from __future__ import annotations

from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cimpy.cimpy_time_analysis.cim_tool_registry import CIMToolRegistry
from cimpy.cimpy_time_analysis.langchain_llm import get_llm


_ALLOWED_REQUEST_MODES = {
    "standard_sv",
    "standard_base",
    "standard_listing",
    "standard_comparison",
    "custom_plan",
    "clarification_needed",
}

_ALLOWED_INTENTS = {
    "historical_analysis",
    "topology_query",
    "asset_lookup",
    "unsupported_cim_request",
}

_ALLOWED_CONFIDENCE = {"high", "medium", "low"}
_ALLOWED_TARGET_KINDS = {"asset", "metric", "topology", "unknown"}


class CIMRequestModeDecision(BaseModel):
    intent: str = Field(description="Allowed values are provided in the prompt.")
    confidence: str = Field(description="Allowed values are provided in the prompt.")
    target_kind: str = Field(description="Allowed values are provided in the prompt.")
    request_mode: str = Field(description="Allowed values are provided in the prompt.")
    safe_to_execute: bool = Field(
        description="True if the workflow can be executed with the currently supported capabilities"
    )
    missing_context: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the decision")


class CIMRequestModeOnlyDecision(BaseModel):
    request_mode: str = Field(description="Allowed values are provided in the prompt.")
    reasoning: str = Field(description="Short explanation of the decision")


class CIMCustomPlanDecision(BaseModel):
    required_steps: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of the planning decision")


class CIMDomainAgent:
    """
    Schlanker LLM-basierter Domain Agent für die CIM-Seite.

    Ziel dieses Stands:
    - LLM-basierte Request-Entscheidung
    - Standardfälle werden deterministisch auf feste Steps gemappt
    - freie Planung nur für Nicht-Standardfälle
    - Planner wählt nur aus expliziten Auswahlmöglichkeiten
    - Comparison-Priorisierung bleibt LLM-basiert, nicht heuristisch
    - robustere Parser-Validierung ohne fragile Literal-Zwänge
    - Registry bleibt die ausführende Schicht
    """

    def __init__(self, cim_root: str):
        self.cim_root = cim_root
        self.llm = get_llm()
        self.registry = CIMToolRegistry(cim_root=cim_root)

        self.mode_parser = PydanticOutputParser(
            pydantic_object=CIMRequestModeDecision
        )
        self.mode_only_parser = PydanticOutputParser(
            pydantic_object=CIMRequestModeOnlyDecision
        )
        self.custom_plan_parser = PydanticOutputParser(
            pydantic_object=CIMCustomPlanDecision
        )

        self.mode_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the request classifier for a CIM domain agent.

Your task:
- classify the request semantically
- decide whether it is safe to execute
- choose exactly one request_mode from the allowed list
- choose intent, confidence and target_kind only from the allowed lists

Allowed intents:
- historical_analysis
- topology_query
- asset_lookup
- unsupported_cim_request

Allowed confidence values:
- high
- medium
- low

Allowed target_kind values:
- asset
- metric
- topology
- unknown

Allowed request modes:
- standard_sv
- standard_base
- standard_listing
- standard_comparison
- custom_plan
- clarification_needed

Critical rules:
- You must choose exactly one request_mode from the allowed list.
- Do not invent request modes.
- Do not invent intent, confidence or target_kind values.
- Do not plan execution steps here.
- Do not reject a request only because a date looks like it is in the future relative to today's calendar date.
- Calendar-based "future date" rejection is forbidden.
- Date feasibility is validated later against available snapshot data during execution.
- A request with an explicit date can still be safe to execute.
- Do NOT mark a request unsupported only because an attribute or value may later turn out to be unavailable on the resolved object.
- Attribute/value validation happens later during execution.
- If the request is unclear or not safely executable for domain reasons, set safe_to_execute=false.
- Keep missing_context specific and real.
- Return only structured output.

Semantic guidance:
- standard_sv: standard dynamic SV/state value request such as power, voltage, P, Q, S or similar state values.
- standard_base: standard static base/nameplate attribute request.
- standard_listing: standard type-listing request asking which objects of a CIM equipment type exist.
- standard_comparison: standard comparison/limit-check request comparing SV values against base values.
- custom_plan: request is CIM-related and executable, but deviates from the standard cases and needs explicit step planning.
- clarification_needed: request is CIM-related, but not safely executable because essential context is missing.

Priority rule:
- When a request contains both a dynamic state quantity and an explicit comparison / utilization / loading / limit-check meaning, choose standard_comparison over standard_sv.
- Comparison semantics take precedence over pure state-value semantics.

Practical guidance:
- Requests about "Leistung", "Power", "Spannung", "Voltage", P, Q, S or similar dynamic state values usually mean standard_sv.
- If such terms are explicitly modified by base-attribute semantics such as "Nenn-", "Bemessungs-", "rated", "nominal", "initial", "max", "min", "operating mode", "technical id", "mRID" or similar static wording, they usually mean standard_base.
- Requests asking which transformers, lines, loads, generators or other CIM equipment objects exist usually mean standard_listing.
- Requests asking for overload, loading, utilization, threshold violations, limit violations, voltage limit checks or a comparison between an SV value and a base value usually mean standard_comparison.
- Requests asking when the highest loading or utilization occurred for a specific asset in a given time window also usually mean standard_comparison.
- "Auslastung" of a transformer is normally standard_comparison, not standard_sv.
- A request such as "Wie war die Spannung ... im Vergleich zu den Spannungsgrenzen?" is standard_comparison, not unsupported.
- A concrete value request for a concrete equipment instance should usually be standard_sv or standard_base, not unsupported.

{format_instructions}
"""
            ),
            ("user", "User request:\n{user_input}")
        ])

        self.mode_only_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a fallback request-mode classifier for a CIM domain agent.

Your only task:
- choose exactly one request_mode from the allowed list

Allowed request modes:
- standard_sv
- standard_base
- standard_listing
- standard_comparison
- custom_plan
- clarification_needed

Critical rules:
- You must choose exactly one request_mode from the allowed list.
- Do not invent request modes.
- Return only structured output.

Priority rule:
- If the request mixes a dynamic state quantity with comparison / utilization / loading / limit-check semantics,
  choose standard_comparison over standard_sv.

Key semantic anchors:
- standard_listing: asks which objects of a CIM equipment type exist
- standard_base: asks for static nameplate / technical base attributes
- standard_sv: asks for a dynamic state value only
- standard_comparison: asks for utilization, loading, overload, threshold/limit checks, or compares a dynamic value against a base/reference/limit
- clarification_needed: essential context is missing

Examples:
- "Was war die Auslastung von Trafo 19-20 am 2026-01-09?" => standard_comparison
- "Wie war die Spannung des Trafos 19-20 am 2026-01-09 im Vergleich zu den Spannungsgrenzen?" => standard_comparison
- "Wie hoch war die Spannung des Trafos 19-20 am 2026-01-09?" => standard_sv
- "Wann war die höchste Auslastung von Trafo 19-20 am 2026-01-09?" => standard_comparison
- "Was ist die Nennspannung von Trafo 19-20?" => standard_base

{format_instructions}
"""
            ),
            ("user", "User request:\n{user_input}")
        ])

        self.custom_plan_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the custom planner for a CIM domain agent.

Your task:
- choose the required internal execution steps for a non-standard but supported CIM request

Available internal tools:
{available_tools_text}

Critical rules:
- You may only choose tool names from the available internal tools list.
- Do not invent tool names.
- Choose only the steps needed for this specific request.
- Do NOT assume a fixed workflow for every request.
- For supported custom flows, summarize_cim_result should usually be the final step.
- Return only structured output.

Practical guidance:
- Use scan_snapshot_inventory whenever structure / network index is needed.
- Use list_equipment_of_type for requests that ask which objects of a CIM equipment type exist.
- Use resolve_cim_object to resolve equipment and parse the user query when a concrete equipment instance is needed.
- Use read_cim_base_values for static equipment base or nameplate attributes.
- Use load_snapshot_cache for historical state / metric questions.
- Use query_cim for the actual domain query.
- Use resolve_cim_comparison before reading SV and base values when the request is a comparison/limit-check.
- Use compare_cim_values only after both SV values and base values are available.
- Use summarize_cim_result at the end.

{format_instructions}
"""
            ),
            ("user", "User request:\n{user_input}")
        ])

    # ------------------------------------------------------------------
    # TOOL DISCOVERY
    # ------------------------------------------------------------------
    def _available_tool_specs(self) -> List[Dict[str, Any]]:
        specs = self.registry.list_tool_specs()
        return specs if isinstance(specs, list) else []

    def _available_tool_names(self) -> Set[str]:
        names: Set[str] = set()

        for spec in self._available_tool_specs():
            name = spec.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())

        names.add("unsupported_request")
        return names

    def _build_available_tools_text(self) -> str:
        lines: List[str] = []

        for spec in self._available_tool_specs():
            name = spec.get("name", "")
            description = spec.get("description", "")
            capability_tags = spec.get("capability_tags", []) or []
            mutating = bool(spec.get("mutating", False))

            tags_text = ", ".join(str(tag) for tag in capability_tags) if capability_tags else "-"
            lines.append(
                f"- {name}: {description} | tags={tags_text} | mutating={mutating}"
            )

        return "\n".join(lines)

    def _standard_required_steps(self, request_mode: str) -> List[str]:
        if request_mode == "standard_listing":
            return ["list_equipment_of_type"]

        if request_mode == "standard_base":
            return ["read_cim_base_values"]

        if request_mode == "standard_sv":
            return ["query_cim"]

        if request_mode == "standard_comparison":
            return [
                "resolve_cim_comparison",
                "query_cim",
                "read_cim_base_values",
                "compare_cim_values",
            ]

        return []

    def _normalize_custom_steps(self, requested_steps: List[Any]) -> List[str]:
        allowed_names = self._available_tool_names()
        if not isinstance(requested_steps, list):
            return []

        return [
            step
            for step in requested_steps
            if isinstance(step, str) and step in allowed_names and step != "unsupported_request"
        ]

    def _clean_enum_value(self, value: Any) -> str:
        return str(value or "").strip()

    def _normalize_mode_result(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        request_mode = self._clean_enum_value(raw.get("request_mode"))
        intent = self._clean_enum_value(raw.get("intent"))
        confidence = self._clean_enum_value(raw.get("confidence"))
        target_kind = self._clean_enum_value(raw.get("target_kind"))

        if request_mode not in _ALLOWED_REQUEST_MODES:
            request_mode = "clarification_needed"

        if intent not in _ALLOWED_INTENTS:
            intent = "historical_analysis"

        if confidence not in _ALLOWED_CONFIDENCE:
            confidence = "medium"

        if target_kind not in _ALLOWED_TARGET_KINDS:
            target_kind = "unknown"

        safe_to_execute = bool(raw.get("safe_to_execute", False))
        missing_context = raw.get("missing_context", [])
        if not isinstance(missing_context, list):
            missing_context = []

        reasoning = str(raw.get("reasoning", "") or "").strip()

        return {
            "intent": intent,
            "confidence": confidence,
            "target_kind": target_kind,
            "request_mode": request_mode,
            "safe_to_execute": safe_to_execute,
            "missing_context": [str(x) for x in missing_context if str(x).strip()],
            "required_steps": [],
            "reasoning": reasoning,
        }

    def _fallback_request_mode_via_llm(self, user_input: str) -> Dict[str, Any]:
        chain = self.mode_only_prompt | self.llm | self.mode_only_parser
        decision = chain.invoke({
            "user_input": user_input,
            "format_instructions": self.mode_only_parser.get_format_instructions(),
        })

        if hasattr(decision, "dict"):
            result = decision.dict()
        else:
            result = dict(decision)

        request_mode = self._clean_enum_value(result.get("request_mode"))
        if request_mode not in _ALLOWED_REQUEST_MODES:
            request_mode = "clarification_needed"

        reasoning = str(result.get("reasoning", "") or "").strip()

        return {
            "intent": "historical_analysis",
            "confidence": "medium",
            "target_kind": "metric",
            "request_mode": request_mode,
            "safe_to_execute": request_mode != "clarification_needed",
            "missing_context": [],
            "required_steps": [],
            "reasoning": f"fallback_llm_mode_classifier: {reasoning}".strip(),
        }

    # ------------------------------------------------------------------
    # PLANNING
    # ------------------------------------------------------------------
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

    def build_plan(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        available_specs = {
            spec["name"]: spec
            for spec in self._available_tool_specs()
            if isinstance(spec, dict) and spec.get("name")
        }

        allowed_names = set(available_specs.keys())
        required_steps = classification.get("required_steps", [])
        request_mode = str(classification.get("request_mode", "") or "").strip()

        if not isinstance(required_steps, list):
            required_steps = []

        safe_to_execute = bool(classification.get("safe_to_execute", False))
        if not safe_to_execute:
            return [{
                "step": "unsupported_request",
                "description": "Return a controlled unsupported-workflow message",
            }]

        normalized_steps: List[str] = []

        def add(step_name: str) -> None:
            if step_name in allowed_names and step_name not in normalized_steps:
                normalized_steps.append(step_name)

        for step in required_steps:
            if step == "unsupported_request":
                return [{
                    "step": "unsupported_request",
                    "description": "Return a controlled unsupported-workflow message",
                }]

            if step == "resolve_cim_comparison":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                add("resolve_cim_comparison")
                continue

            if step == "query_cim":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                if "resolve_cim_comparison" in required_steps:
                    add("resolve_cim_comparison")
                add("load_snapshot_cache")
                add("query_cim")
                continue

            if step == "load_snapshot_cache":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                add("load_snapshot_cache")
                continue

            if step == "resolve_cim_object":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                continue

            if step == "list_equipment_of_type":
                add("scan_snapshot_inventory")
                add("list_equipment_of_type")
                continue

            if step == "read_cim_base_values":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                if "resolve_cim_comparison" in required_steps:
                    add("resolve_cim_comparison")
                add("read_cim_base_values")
                continue

            if step == "compare_cim_values":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                add("resolve_cim_comparison")
                add("load_snapshot_cache")
                add("query_cim")
                add("read_cim_base_values")
                add("compare_cim_values")
                continue

            if step == "scan_snapshot_inventory":
                add("scan_snapshot_inventory")
                continue

            if step == "summarize_cim_result":
                add("summarize_cim_result")
                continue

            if step in allowed_names:
                add(step)

        if not normalized_steps:
            if request_mode == "standard_listing":
                add("scan_snapshot_inventory")
                add("list_equipment_of_type")
            elif request_mode == "standard_base":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                add("read_cim_base_values")
                add("summarize_cim_result")
            elif request_mode == "standard_sv":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                add("load_snapshot_cache")
                add("query_cim")
                add("summarize_cim_result")
            elif request_mode == "standard_comparison":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
                add("resolve_cim_comparison")
                add("load_snapshot_cache")
                add("query_cim")
                add("read_cim_base_values")
                add("compare_cim_values")
                add("summarize_cim_result")
            else:
                intent = classification.get("intent", "")

                if intent in {"historical_analysis", "topology_query", "asset_lookup"}:
                    add("scan_snapshot_inventory")
                    add("resolve_cim_object")
                    if intent == "historical_analysis":
                        add("load_snapshot_cache")
                    add("query_cim")
                    add("summarize_cim_result")
                else:
                    return [{
                        "step": "unsupported_request",
                        "description": "Return a controlled unsupported-workflow message",
                    }]

        if (
            "summarize_cim_result" in allowed_names
            and "summarize_cim_result" not in normalized_steps
            and "list_equipment_of_type" not in normalized_steps
        ):
            normalized_steps.append("summarize_cim_result")

        plan: List[Dict[str, Any]] = []

        for step in normalized_steps:
            spec = available_specs.get(step)
            if not spec:
                continue

            plan.append({
                "step": step,
                "description": spec.get("description", ""),
                "capability_tags": spec.get("capability_tags", []) or [],
                "mutating": bool(spec.get("mutating", False)),
            })

        if not plan:
            plan = [{
                "step": "unsupported_request",
                "description": "Return a controlled unsupported-workflow message",
            }]

        return plan

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def build_unsupported_result(
        self,
        user_input: str,
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        missing_context = classification.get("missing_context", []) or []
        missing_text = ""

        if missing_context:
            missing_text = " Fehlender Kontext: " + ", ".join(missing_context) + "."

        return {
            "status": "error",
            "tool": "cim_agent",
            "agent": "CIMDomainAgent",
            "error": "unsupported_cim_request",
            "answer": (
                "Die Anfrage wurde zur CIM-Seite geroutet, ist aktuell aber noch nicht "
                "sicher oder unterstützt ausführbar."
                + missing_text
            ),
            "classification": classification,
            "user_input": user_input,
        }

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    def execute_plan(
        self,
        user_input: str,
        classification: Dict[str, Any],
        plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "user_input": user_input,
            "cim_root": self.cim_root,
            "classification": classification,
        }

        trace: List[Dict[str, Any]] = []

        for item in plan:
            step = item["step"]

            if step == "unsupported_request":
                result = self.build_unsupported_result(
                    user_input=user_input,
                    classification=classification,
                )
                trace.append({
                    "step": step,
                    "result": result,
                })
                result["debug"] = {
                    "trace": trace,
                    "classification": classification,
                    "plan": plan,
                }
                return result

            tool_spec = self.registry.get_tool_spec(step)
            result = self.registry.invoke(step, context=context)

            trace.append({
                "step": step,
                "tool_spec": {
                    "name": tool_spec.name if tool_spec else step,
                    "description": tool_spec.description if tool_spec else "",
                    "capability_tags": tool_spec.capability_tags if tool_spec else [],
                    "mutating": tool_spec.mutating if tool_spec else False,
                },
                "result": result,
            })

            if result.get("status") != "ok":
                result["agent"] = "CIMDomainAgent"
                result["debug"] = {
                    "trace": trace,
                    "classification": classification,
                    "plan": plan,
                }
                return result

            context.update(result)

        return {
            "status": "ok",
            "tool": "cim_agent",
            "agent": "CIMDomainAgent",
            "classification": classification,
            "plan": plan,
            "available_tools": self.registry.list_tool_specs(),
            "answer": context.get("answer", ""),
            "debug": {
                "trace": trace,
            },
        }

    # ------------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        classification = self.classify_request(user_input)
        plan = self.build_plan(classification)

        return self.execute_plan(
            user_input=user_input,
            classification=classification,
            plan=plan,
        )
