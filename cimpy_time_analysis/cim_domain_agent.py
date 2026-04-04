from __future__ import annotations

from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cimpy.cimpy_time_analysis.cim_tool_registry import CIMToolRegistry
from cimpy.cimpy_time_analysis.langchain_llm import get_llm



def _looks_like_type_listing_request(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return False

    listing_markers = [
        "welche",
        "welcher",
        "welches",
        "gibt es",
        "zeige alle",
        "liste alle",
        "alle ",
        "list all",
        "show all",
        "which",
        "what",
    ]
    metric_markers = [
        "spannung",
        "voltage",
        "leistung",
        "power",
        "wirkleistung",
        "blindleistung",
        "scheinleistung",
        "auslastung",
        "loading",
        "nachbarn",
        "neighbors",
        "connected",
        "komponente",
        "path",
        "pfad",
        "über die zeit",
        "over time",
    ]

    has_listing = any(marker in text for marker in listing_markers)
    has_metric_or_topology = any(marker in text for marker in metric_markers)
    return has_listing and not has_metric_or_topology


class CIMPlannerDecision(BaseModel):
    intent: str = Field(
        description="One of: historical_analysis, topology_query, asset_lookup, unsupported_cim_request"
    )
    confidence: str = Field(
        description="One of: high, medium, low"
    )
    target_kind: str = Field(
        description="asset, metric, topology, unknown"
    )
    safe_to_execute: bool = Field(
        description="True if the workflow can be executed with the currently supported capabilities"
    )
    missing_context: List[str] = Field(default_factory=list)
    required_steps: List[str] = Field(default_factory=list)
    reasoning: str = Field(
        description="Short explanation of the planning decision"
    )


class CIMDomainAgent:
    """
    Schlanker LLM-basierter Domain Agent für die CIM-Seite.

    Ziel dieses Stands:
    - aufgabenabhängige Planung durch den LLM
    - keine starre Intent->Workflow-Verdrahtung
    - minimale technische Plan-Normalisierung
    - Registry bleibt die ausführende Schicht
    """

    def __init__(self, cim_root: str):
        self.cim_root = cim_root
        self.llm = get_llm()
        self.registry = CIMToolRegistry(cim_root=cim_root)

        self.parser = PydanticOutputParser(
            pydantic_object=CIMPlannerDecision
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the planner for a CIM domain agent.

Your task:
- classify the request
- decide whether it is safe to execute
- choose the required internal execution steps from the available tool list

Supported intents:
- historical_analysis
- topology_query
- asset_lookup
- unsupported_cim_request

Available internal tools:
{available_tools_text}

Critical rules:
- Only use tool names from the available internal tools list.
- Choose only the steps needed for this specific request.
- Do NOT assume a fixed workflow for every request.
- Do NOT reject a request only because a date looks like it is in the future relative to today's calendar date.
- Calendar-based "future date" rejection is forbidden.
- Date feasibility is validated later against available snapshot data during execution.
- A request with an explicit date can still be safe to execute.
- If the request is unclear or not safely executable for domain reasons, set safe_to_execute=false.
- In that case, use required_steps=["unsupported_request"] and fill missing_context only with real missing information.
- For supported flows, summarize_cim_result should usually be the final step.
- Return only structured output.

Practical guidance:
- Use scan_snapshot_inventory whenever structure / network index is needed.
- Use list_equipment_of_type for requests that ask which objects of a CIM equipment type exist, for example list/show/which/all requests about transformers, lines, loads or generators.
- Use resolve_cim_object to resolve equipment and parse the user query when a concrete equipment instance is needed.
- Use load_snapshot_cache for historical state / metric questions.
- Use query_cim for the actual domain query.
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

        lines.append("- unsupported_request: Return a controlled unsupported-workflow message")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PLANNING
    # ------------------------------------------------------------------
    def classify_request(self, user_input: str) -> Dict[str, Any]:
        try:
            chain = self.prompt | self.llm | self.parser

            decision = chain.invoke({
                "user_input": user_input,
                "available_tools_text": self._build_available_tools_text(),
                "format_instructions": self.parser.get_format_instructions(),
            })

            if hasattr(decision, "dict"):
                result = decision.dict()
            else:
                result = dict(decision)

            allowed_names = self._available_tool_names()
            requested_steps = result.get("required_steps", [])

            if not isinstance(requested_steps, list):
                requested_steps = []

            result["required_steps"] = [
                step
                for step in requested_steps
                if isinstance(step, str) and step in allowed_names
            ]

            if _looks_like_type_listing_request(user_input) and "list_equipment_of_type" in allowed_names:
                result["safe_to_execute"] = True
                result["required_steps"] = ["list_equipment_of_type"]
                reasoning = str(result.get("reasoning", "") or "").strip()
                heuristic_reason = "type-listing heuristic selected list_equipment_of_type"
                result["reasoning"] = f"{reasoning} | {heuristic_reason}" if reasoning else heuristic_reason

            if not result["required_steps"] and result.get("safe_to_execute", False):
                result["safe_to_execute"] = False
                result["missing_context"] = list(result.get("missing_context", []) or [])
                result["required_steps"] = ["unsupported_request"]
                if "planner returned no executable steps" not in result["missing_context"]:
                    result["missing_context"].append("planner returned no executable steps")

            return {
                "status": "ok",
                "classification_mode": "llm",
                **result,
            }

        except Exception as exc:
            return {
                "status": "ok",
                "classification_mode": "fallback",
                "intent": "unsupported_cim_request",
                "confidence": "low",
                "target_kind": "unknown",
                "safe_to_execute": False,
                "missing_context": [],
                "required_steps": ["unsupported_request"],
                "reasoning": f"planner_error: {exc}",
            }

    def build_plan(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        available_specs = {
            spec["name"]: spec
            for spec in self._available_tool_specs()
            if isinstance(spec, dict) and spec.get("name")
        }

        allowed_names = set(available_specs.keys())
        required_steps = classification.get("required_steps", [])
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

            if step == "query_cim":
                add("scan_snapshot_inventory")
                add("resolve_cim_object")
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

            if step == "scan_snapshot_inventory":
                add("scan_snapshot_inventory")
                continue

            if step == "summarize_cim_result":
                add("summarize_cim_result")
                continue

            if step in allowed_names:
                add(step)

        if not normalized_steps:
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