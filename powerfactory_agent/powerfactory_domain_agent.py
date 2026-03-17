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
        description="One of: load_catalog, change_load, topology_query, unsupported_powerfactory_request"
    )
    confidence: str = Field(
        description="One of: high, medium, low"
    )
    target_kind: str = Field(
        description="Main target type, e.g. load, topology_asset, catalog, unknown"
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
            "interpret_instruction, resolve_load, execute_change_load, summarize_powerfactory_result, "
            "build_topology_graph, build_topology_inventory, interpret_entity_instruction, "
            "resolve_entity_from_inventory, query_topology_neighbors, summarize_topology_result, "
            "unsupported_request"
        )
    )
    reasoning: str = Field(
        description="Short explanation of why this plan was selected"
    )


class PowerFactoryDomainAgent:
    def __init__(self, project_name: str = DEFAULT_PROJECT_NAME):
        self.project_name = project_name

        self.llm = get_llm()
        self.registry = PowerFactoryToolRegistry()

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
                "- topology_query: user wants to inspect the network topology, e.g. neighbors or connectivity around an asset.\n"
                "- unsupported_powerfactory_request: the request is routed to PowerFactory "
                "but does not fit the currently supported workflows.\n\n"
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
                "- unsupported_request\n\n"
                "Rules:\n"
                "- Only produce supported intents.\n"
                "- Only use allowed required_steps.\n"
                "- If the request is about listing or discovering loads, use load_catalog.\n"
                "- If the request is about changing a load, use change_load.\n"
                "- If the request is about topology, neighboring assets, connected components or connectivity around an asset, use topology_query.\n"
                "- For topology_query, prefer a staged resolution process over direct free-text matching.\n"
                "- If the request does not fit supported PowerFactory workflows, use unsupported_powerfactory_request.\n"
                "- If the request is unsafe or not executable with current capabilities, set safe_to_execute=false.\n"
                "- If information is missing, list it in missing_context.\n"
                "- For load_catalog, usually required_steps should be:\n"
                "  [get_load_catalog, summarize_load_catalog]\n"
                "- For change_load, usually required_steps should be:\n"
                "  [interpret_instruction, resolve_load, execute_change_load, summarize_powerfactory_result]\n"
                "- For topology_query, usually required_steps should be:\n"
                "  [build_topology_graph, build_topology_inventory, interpret_entity_instruction, resolve_entity_from_inventory, query_topology_neighbors, summarize_topology_result]\n"
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

    def build_planner_chain(self):
        return self.planner_prompt | self.llm | self.planner_parser

    def classify_request(self, user_input: str) -> Dict[str, Any]:
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
        allowed_steps = {
            "get_load_catalog": "Read the available load catalog from the active PowerFactory project",
            "summarize_load_catalog": "Build a concise user-facing catalog answer",
            "interpret_instruction": "Interpret user input into structured PowerFactory load instruction",
            "resolve_load": "Resolve the target load inside the active PowerFactory project",
            "execute_change_load": "Apply the requested load change and run load flow before/after",
            "summarize_powerfactory_result": "Interpret the result data and generate final user answer",
            "build_topology_graph": "Build a PowerFactory topology graph from the active project",
            "build_topology_inventory": "Build a typed inventory from the current PowerFactory topology graph",
            "interpret_entity_instruction": "Interpret user input into a structured generic PowerFactory entity instruction",
            "resolve_entity_from_inventory": "Resolve the requested entity from the PowerFactory topology inventory",
            "query_topology_neighbors": "Query neighboring assets in the PowerFactory topology graph",
            "summarize_topology_result": "Build a concise user-facing topology answer",
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

        return [
            {
                "step": "unsupported_request",
                "description": allowed_steps["unsupported_request"],
            }
        ]

    def build_plan(self, user_input: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.normalize_required_steps(classification)

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
        graph_summary = graph_result.get("graph_summary", {}) if isinstance(graph_result, dict) else {}
        build_debug = graph_result.get("build_debug", {}) if isinstance(graph_result, dict) else {}
        inventory = inventory_result.get("inventory", {}) if isinstance(inventory_result, dict) else {}

        selected_name = selected.get("name") or selected.get("full_name") or "<unbekannt>"
        selected_class = selected.get("pf_class") or "<unknown>"
        selected_type = selected.get("inventory_type") or "<unknown>"
        neighbor_count = topology_result.get("neighbor_count", len(neighbors))

        resolved_asset_query = None
        if isinstance(entity_resolution, dict):
            resolved_asset_query = entity_resolution.get("asset_query")

        if neighbor_count == 0:
            answer = (
                f"Für das Asset '{selected_name}' ({selected_class}) wurden im aktuellen "
                "PowerFactory-Topologiegraphen keine direkten Nachbarn gefunden."
            )
        else:
            preview_items = []
            for neighbor in neighbors[:10]:
                neighbor_name = neighbor.get("name") or neighbor.get("full_name") or "<unbekannt>"
                neighbor_class = neighbor.get("pf_class") or "<unknown>"
                preview_items.append(f"{neighbor_name} ({neighbor_class})")

            if neighbor_count <= 10:
                answer = (
                    f"Direkte Nachbarn von '{selected_name}' ({selected_class}, Typ {selected_type}) "
                    f"im PowerFactory-Topologiegraphen: " + ", ".join(preview_items)
                )
            else:
                answer = (
                    f"Für '{selected_name}' ({selected_class}, Typ {selected_type}) wurden {neighbor_count} direkte Nachbarn "
                    f"im PowerFactory-Topologiegraphen gefunden. Beispiele: "
                    + ", ".join(preview_items)
                )

        return {
            "status": "ok",
            "tool": "summarize_topology_result",
            "answer": answer,
            "selected_node": selected,
            "neighbor_count": neighbor_count,
            "neighbors": neighbors,
            "graph_summary": graph_summary,
            "build_debug": build_debug,
            "inventory_types": inventory.get("available_types", []),
            "resolved_asset_query": resolved_asset_query,
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
            "answer": (
                "Die Anfrage wurde nach PowerFactory geroutet, passt aber aktuell zu keinem "
                "unterstützten PowerFactory-Ablauf oder ist noch nicht sicher ausführbar."
                + missing_text
            ),
            "user_input": user_input,
            "classification": classification,
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return self.registry.list_tool_specs()

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

        graph_result = None
        inventory_result = None
        entity_instruction = None
        entity_resolution = None
        topology_result = None

        for item in plan:
            step = item["step"]

            if step == "summarize_load_catalog":
                result = self.summarize_load_catalog_result(catalog_result)
                debug_trace.append({"step": step, "result": result})

                if result["status"] != "ok":
                    return self.build_error_result(error_result=result, debug_trace=debug_trace)

                summary = result
                continue

            if step == "summarize_topology_result":
                result = self.summarize_topology_result(
                    topology_result=topology_result,
                    graph_result=graph_result,
                    inventory_result=inventory_result,
                    entity_instruction=entity_instruction,
                    entity_resolution=entity_resolution,
                )
                debug_trace.append({"step": step, "result": result})

                if result["status"] != "ok":
                    return self.build_error_result(error_result=result, debug_trace=debug_trace)

                summary = result
                continue

            if step == "unsupported_request":
                result = self.build_unsupported_result(
                    user_input=user_input,
                    classification=classification,
                )
                debug_trace.append({"step": step, "result": result})
                return result

            tool_kwargs: Dict[str, Any] = {
                "services": services,
            }

            if step == "interpret_instruction":
                tool_kwargs["user_input"] = user_input
            elif step == "resolve_load":
                tool_kwargs["instruction"] = instruction
            elif step == "execute_change_load":
                tool_kwargs["instruction"] = instruction
            elif step == "summarize_powerfactory_result":
                tool_kwargs["result_payload"] = execution
                tool_kwargs["user_input"] = user_input
            elif step == "build_topology_graph":
                tool_kwargs["contract_cubicles"] = True
            elif step == "build_topology_inventory":
                tool_kwargs["topology_graph_result"] = graph_result
            elif step == "interpret_entity_instruction":
                tool_kwargs["user_input"] = user_input
                tool_kwargs["inventory"] = inventory_result["inventory"] if isinstance(inventory_result, dict) else {}
            elif step == "resolve_entity_from_inventory":
                tool_kwargs["instruction"] = entity_instruction
                tool_kwargs["inventory"] = inventory_result["inventory"] if isinstance(inventory_result, dict) else {}
                tool_kwargs["topology_graph"] = graph_result["topology_graph"] if isinstance(graph_result, dict) else None
                tool_kwargs["max_matches"] = 10
            elif step == "query_topology_neighbors":
                tool_kwargs["topology_graph"] = graph_result["topology_graph"] if isinstance(graph_result, dict) else None
                tool_kwargs["asset_query"] = (
                    entity_resolution.get("asset_query")
                    if isinstance(entity_resolution, dict)
                    else user_input
                )
                tool_kwargs["selected_node_id"] = (
                    entity_resolution.get("selected_match", {}) or {}
                ).get("node_id") if isinstance(entity_resolution, dict) else None
                tool_kwargs["matches"] = entity_resolution.get("matches", []) if isinstance(entity_resolution, dict) else []
                tool_kwargs["max_matches"] = 10

            tool_spec = self.registry.get_tool_spec(step)
            result = self.registry.invoke(step, **tool_kwargs)

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

            if result["status"] != "ok":
                return self.build_error_result(error_result=result, debug_trace=debug_trace)

            if step == "get_load_catalog":
                catalog_result = result
            elif step == "interpret_instruction":
                instruction = result["instruction"]
            elif step == "resolve_load":
                resolution = result
            elif step == "execute_change_load":
                execution = result
            elif step == "summarize_powerfactory_result":
                summary = result
            elif step == "build_topology_graph":
                graph_result = result
            elif step == "build_topology_inventory":
                inventory_result = result
            elif step == "interpret_entity_instruction":
                entity_instruction = result["instruction"]
            elif step == "resolve_entity_from_inventory":
                entity_resolution = result
            elif step == "query_topology_neighbors":
                topology_result = result

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
            graph_result=graph_result,
            inventory_result=inventory_result,
            entity_instruction=entity_instruction,
            entity_resolution=entity_resolution,
            topology_result=topology_result,
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
        catalog_result: Dict[str, Any] | None,
        graph_result: Dict[str, Any] | None,
        inventory_result: Dict[str, Any] | None,
        entity_instruction: Dict[str, Any] | None,
        entity_resolution: Dict[str, Any] | None,
        topology_result: Dict[str, Any] | None,
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
            "available_tools": self.get_available_tools(),
            "instruction": instruction,
            "resolved_load": execution.get("resolved_load") if isinstance(execution, dict) else None,
            "data": execution.get("data", {}) if isinstance(execution, dict) else {},
            "catalog": catalog_result.get("loads", []) if isinstance(catalog_result, dict) else [],
            "messages": summary.get("messages", []) if isinstance(summary, dict) else [],
            "answer": answer,
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
            "debug": {
                "resolution": resolution,
                "execution": execution,
                "summary": summary,
                "graph_result": graph_result,
                "inventory_result": inventory_result,
                "entity_instruction": entity_instruction,
                "entity_resolution": entity_resolution,
                "topology_result": topology_result,
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
        result["available_tools"] = self.get_available_tools()
        result["debug"] = {
            "trace": debug_trace,
        }
        return result

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

    def get_load_catalog(self) -> Dict[str, Any]:
        services = build_powerfactory_services(project_name=self.project_name)
        if services["status"] != "ok":
            return services
        return _get_load_catalog_from_services(services)