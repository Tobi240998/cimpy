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
    _interpret_instruction_with_services,
    _resolve_load_with_services,
    _execute_change_load_with_services,
    _summarize_powerfactory_result_with_services,
)


class PFPlannerDecision(BaseModel):
    intent: str = Field(
        description="One of: load_catalog, change_load, unsupported_powerfactory_request"
    )
    confidence: str = Field(
        description="One of: high, medium, low"
    )
    target_kind: str = Field(
        description="Main target type, e.g. load, catalog, unknown"
    )
    safe_to_execute: bool = Field(
        description="True if the request can be executed with the currently supported workflow"
    )
    missing_context: List[str] = Field(
        default_factory=list,
        description="List of missing information items if the request is underspecified"
    )
    required_steps: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of required internal steps. Allowed values are: "
            "get_load_catalog, summarize_load_catalog, "
            "interpret_instruction, resolve_load, execute_change_load, "
            "summarize_powerfactory_result, unsupported_request"
        )
    )
    reasoning: str = Field(
        description="Short explanation of why this plan was selected"
    )


class PowerFactoryDomainAgent:
    """
    Domain agent responsible for:
    - planning the internal PowerFactory workflow
    - calling PowerFactory subtools
    - returning a structured result with debug trace
    """

    def __init__(self, project_name: str = DEFAULT_PROJECT_NAME):
        self.project_name = project_name

        self.llm = get_llm()
        self.planner_parser = PydanticOutputParser(
            pydantic_object=PFPlannerDecision
        )

        self.planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a planning assistant for a PowerFactory domain agent.\n"
                "Your task is to classify the request and produce a safe internal plan.\n\n"
                "Supported intents:\n"
                "- load_catalog: user wants to inspect or list available loads.\n"
                "- change_load: user wants to increase, decrease, set or modify a load.\n"
                "- unsupported_powerfactory_request: the request is routed to PowerFactory "
                "but does not fit the currently supported workflows.\n\n"
                "Allowed required_steps values:\n"
                "- get_load_catalog\n"
                "- summarize_load_catalog\n"
                "- interpret_instruction\n"
                "- resolve_load\n"
                "- execute_change_load\n"
                "- summarize_powerfactory_result\n"
                "- unsupported_request\n\n"
                "Rules:\n"
                "- Only produce supported intents.\n"
                "- Only use allowed required_steps.\n"
                "- If the request is about listing or discovering loads, use load_catalog.\n"
                "- If the request is about changing a load, use change_load.\n"
                "- If the request does not fit supported PowerFactory workflows, use unsupported_powerfactory_request.\n"
                "- If the request is unsafe or not executable with current capabilities, set safe_to_execute=false.\n"
                "- If information is missing, list it in missing_context.\n"
                "- For load_catalog, usually required_steps should be:\n"
                "  [get_load_catalog, summarize_load_catalog]\n"
                "- For change_load, usually required_steps should be:\n"
                "  [interpret_instruction, resolve_load, execute_change_load, summarize_powerfactory_result]\n"
                "- For unsupported_powerfactory_request, use:\n"
                "  [unsupported_request]\n\n"
                "Return only structured output.\n\n"
                "{format_instructions}"
            ),
            (
                "user",
                "User request:\n{user_input}"
            ),
        ])

    # ------------------------------------------------------------------
    # LLM PLANNING PART
    # ------------------------------------------------------------------
    def build_planner_chain(self):
        return self.planner_prompt | self.llm | self.planner_parser

    def classify_request(self, user_input: str) -> Dict[str, Any]:
        """
        LLM-based request classification and planning.
        """
        try:
            chain = self.build_planner_chain()
            decision = chain.invoke({
                "user_input": user_input,
                "format_instructions": self.planner_parser.get_format_instructions(),
            })

            if hasattr(decision, "dict"):
                result = decision.dict()
            else:
                result = dict(decision)

            return {
                "status": "ok",
                "classification_mode": "llm",
                **result,
            }
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

    def normalize_required_steps(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validates and normalizes planner output into internal step objects.
        """
        allowed_steps = {
            "get_load_catalog": "Read the available load catalog from the active PowerFactory project",
            "summarize_load_catalog": "Build a concise user-facing catalog answer",
            "interpret_instruction": "Interpret user input into structured PowerFactory instruction",
            "resolve_load": "Resolve the target load inside the active PowerFactory project",
            "execute_change_load": "Apply the requested load change and run load flow before/after",
            "summarize_powerfactory_result": "Interpret the result data and generate final user answer",
            "unsupported_request": "Return a controlled message for unsupported PowerFactory intent",
        }

        raw_steps = classification.get("required_steps", [])
        intent = classification.get("intent")
        safe_to_execute = classification.get("safe_to_execute", False)

        if not isinstance(raw_steps, list):
            raw_steps = []

        normalized = []
        for step in raw_steps:
            if step in allowed_steps:
                normalized.append({
                    "step": step,
                    "description": allowed_steps[step],
                })

        if normalized:
            return normalized

        if intent == "load_catalog":
            return [
                {
                    "step": "get_load_catalog",
                    "description": allowed_steps["get_load_catalog"],
                },
                {
                    "step": "summarize_load_catalog",
                    "description": allowed_steps["summarize_load_catalog"],
                },
            ]

        if intent == "change_load" and safe_to_execute:
            return [
                {
                    "step": "interpret_instruction",
                    "description": allowed_steps["interpret_instruction"],
                },
                {
                    "step": "resolve_load",
                    "description": allowed_steps["resolve_load"],
                },
                {
                    "step": "execute_change_load",
                    "description": allowed_steps["execute_change_load"],
                },
                {
                    "step": "summarize_powerfactory_result",
                    "description": allowed_steps["summarize_powerfactory_result"],
                },
            ]

        return [
            {
                "step": "unsupported_request",
                "description": allowed_steps["unsupported_request"],
            }
        ]

    # ------------------------------------------------------------------
    # PLANNING PART
    # ------------------------------------------------------------------
    def build_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds an internal execution plan from structured planner output.
        """
        return self.normalize_required_steps(classification)

    # ------------------------------------------------------------------
    # HELPER PART
    # ------------------------------------------------------------------
    def summarize_load_catalog_result(self, catalog_result: Dict[str, Any]) -> Dict[str, Any]:
        loads = catalog_result.get("loads", [])

        names = [entry.get("loc_name") for entry in loads if entry.get("loc_name")]
        preview = names[:10]

        if not names:
            answer = "Im aktiven PowerFactory-Projekt wurden keine Lasten gefunden."
        elif len(names) <= 10:
            answer = "Verfügbare Lasten im aktiven PowerFactory-Projekt: " + ", ".join(names)
        else:
            answer = (
                "Im aktiven PowerFactory-Projekt wurden "
                f"{len(names)} Lasten gefunden. Beispiele: "
                + ", ".join(preview)
            )

        return {
            "status": "ok",
            "tool": "summarize_load_catalog",
            "answer": answer,
            "count": len(names),
            "loads": loads,
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
            "answer": (
                "Die Anfrage wurde nach PowerFactory geroutet, passt aber aktuell zu keinem "
                "unterstützten PowerFactory-Ablauf oder ist noch nicht sicher ausführbar."
                + missing_text
            ),
            "user_input": user_input,
            "classification": classification,
        }

    # ------------------------------------------------------------------
    # EXECUTION PART
    # ------------------------------------------------------------------
    def execute_plan(
        self,
        services: Dict[str, Any],
        plan: List[Dict[str, Any]],
        user_input: str,
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        debug_trace: List[Dict[str, Any]] = []

        instruction = None
        resolution = None
        execution = None
        summary = None
        catalog_result = None

        for item in plan:
            step = item["step"]

            if step == "get_load_catalog":
                result = _get_load_catalog_from_services(services)
                debug_trace.append({
                    "step": step,
                    "result": result,
                })

                if result["status"] != "ok":
                    return self.build_error_result(
                        error_result=result,
                        debug_trace=debug_trace,
                    )

                catalog_result = result

            elif step == "summarize_load_catalog":
                result = self.summarize_load_catalog_result(catalog_result)
                debug_trace.append({
                    "step": step,
                    "result": result,
                })

                if result["status"] != "ok":
                    return self.build_error_result(
                        error_result=result,
                        debug_trace=debug_trace,
                    )

                summary = result

            elif step == "interpret_instruction":
                result = _interpret_instruction_with_services(
                    services=services,
                    user_input=user_input,
                )
                debug_trace.append({
                    "step": step,
                    "result": result,
                })

                if result["status"] != "ok":
                    return self.build_error_result(
                        error_result=result,
                        debug_trace=debug_trace,
                    )

                instruction = result["instruction"]

            elif step == "resolve_load":
                result = _resolve_load_with_services(
                    services=services,
                    instruction=instruction,
                )
                debug_trace.append({
                    "step": step,
                    "result": result,
                })

                if result["status"] != "ok":
                    return self.build_error_result(
                        error_result=result,
                        debug_trace=debug_trace,
                    )

                resolution = result

            elif step == "execute_change_load":
                result = _execute_change_load_with_services(
                    services=services,
                    instruction=instruction,
                )
                debug_trace.append({
                    "step": step,
                    "result": result,
                })

                if result["status"] != "ok":
                    return self.build_error_result(
                        error_result=result,
                        debug_trace=debug_trace,
                    )

                execution = result

            elif step == "summarize_powerfactory_result":
                result = _summarize_powerfactory_result_with_services(
                    services=services,
                    result_payload=execution,
                    user_input=user_input,
                )
                debug_trace.append({
                    "step": step,
                    "result": result,
                })

                if result["status"] != "ok":
                    return self.build_error_result(
                        error_result=result,
                        debug_trace=debug_trace,
                    )

                summary = result

            elif step == "unsupported_request":
                result = self.build_unsupported_result(
                    user_input=user_input,
                    classification=classification,
                )
                debug_trace.append({
                    "step": step,
                    "result": result,
                })
                return result

            else:
                result = {
                    "status": "error",
                    "tool": "powerfactory_domain_agent",
                    "error": f"Unknown plan step: {step}",
                }
                debug_trace.append({
                    "step": step,
                    "result": result,
                })
                return self.build_error_result(
                    error_result=result,
                    debug_trace=debug_trace,
                )

        return self.build_success_result(
            services=services,
            user_input=user_input,
            classification=classification,
            plan=plan,
            instruction=instruction,
            resolution=resolution,
            execution=execution,
            summary=summary,
            catalog_result=catalog_result,
            debug_trace=debug_trace,
        )

    # ------------------------------------------------------------------
    # RESULT PART
    # ------------------------------------------------------------------
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
        catalog_result: Dict[str, Any] | None,
        debug_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer = ""
        if isinstance(summary, dict):
            answer = summary.get("answer", "")

        return {
            "status": "ok",
            "tool": "powerfactory",
            "agent": "PowerFactoryDomainAgent",
            "project": services.get("project_name"),
            "studycase": execution.get("studycase") if isinstance(execution, dict) else None,
            "user_input": user_input,
            "classification": classification,
            "plan": plan,
            "instruction": instruction,
            "resolved_load": execution.get("resolved_load") if isinstance(execution, dict) else None,
            "data": execution.get("data", {}) if isinstance(execution, dict) else {},
            "catalog": catalog_result.get("loads", []) if isinstance(catalog_result, dict) else [],
            "messages": summary.get("messages", []) if isinstance(summary, dict) else [],
            "answer": answer,
            "debug": {
                "resolution": resolution,
                "execution": execution,
                "summary": summary,
                "trace": debug_trace,
            },
        }

    def build_error_result(
        self,
        error_result: Dict[str, Any],
        debug_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = dict(error_result)
        result["agent"] = "PowerFactoryDomainAgent"
        result["debug"] = {
            "trace": debug_trace,
        }
        return result

    # ------------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            return services

        classification = self.classify_request(user_input)
        plan = self.build_plan(
            user_input=user_input,
            classification=classification,
        )

        return self.execute_plan(
            services=services,
            plan=plan,
            user_input=user_input,
            classification=classification,
        )

    # ------------------------------------------------------------------
    # OPTIONAL DISCOVERY
    # ------------------------------------------------------------------
    def get_load_catalog(self) -> Dict[str, Any]:
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            return services
        return _get_load_catalog_from_services(services)