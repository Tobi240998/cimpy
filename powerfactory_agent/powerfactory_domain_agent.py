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
            "unsupported_request"
        )
    )
    reasoning: str = Field(description="Short explanation of why this plan was selected")


class PowerFactoryDomainAgent:
    def __init__(self, project_name: str = DEFAULT_PROJECT_NAME):
        self.project_name = project_name
        self.llm = get_llm()
        self.registry = PowerFactoryToolRegistry()
        self.planner_parser = PydanticOutputParser(pydantic_object=PFPlannerDecision)

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

    def build_planner_chain(self):
        return self.planner_prompt | self.llm | self.planner_parser

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

    def normalize_required_steps(self, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        allowed_steps = {
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
            "build_data_inventory": "Build a lightweight typed inventory for PowerFactory data queries without a topology graph",
            "interpret_data_query_instruction": "Interpret user input into a structured PowerFactory data query instruction",
            "resolve_pf_object_from_inventory_llm": "Resolve the requested PowerFactory object via LLM-based exact candidate selection from the typed inventory",
            "list_available_object_attributes": "Build the available attribute option list for the resolved PowerFactory object",
            "select_pf_object_attributes_llm": "Select matching attributes for the resolved PowerFactory object via LLM from the available attribute option list",
            "read_pf_object_attributes": "Read the selected attributes from the resolved PowerFactory object",
            "summarize_pf_object_data_result": "Summarize the data query result for the user",
            "unsupported_request": "Return a controlled message for unsupported PowerFactory intent",
        }

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

        switch_instruction = None
        switch_resolution = None
        switch_execution = None
        switch_summary = None

        data_inventory_result = None
        data_instruction = None
        data_resolution = None
        data_attribute_listing = None
        data_attribute_selection = None
        data_query_result = None
        data_summary = None

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

            if step == "summarize_pf_object_data_result":
                result = self.registry.invoke(step, services=services, result_payload=data_query_result, user_input=user_input)
                debug_trace.append({"step": step, "result": result})
                if result["status"] != "ok":
                    return self.build_error_result(error_result=result, debug_trace=debug_trace)
                data_summary = result
                summary = result
                continue

            if step == "unsupported_request":
                result = self.build_unsupported_result(user_input=user_input, classification=classification)
                debug_trace.append({"step": step, "result": result})
                return result

            tool_kwargs: Dict[str, Any] = {"services": services}

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
                tool_kwargs["asset_query"] = entity_resolution.get("asset_query") if isinstance(entity_resolution, dict) else user_input
                tool_kwargs["selected_node_id"] = (entity_resolution.get("selected_match", {}) or {}).get("node_id") if isinstance(entity_resolution, dict) else None
                tool_kwargs["matches"] = entity_resolution.get("matches", []) if isinstance(entity_resolution, dict) else []
                tool_kwargs["max_matches"] = 10
            elif step == "interpret_switch_instruction":
                tool_kwargs["user_input"] = user_input
            elif step == "resolve_switch_from_inventory_llm":
                tool_kwargs["instruction"] = switch_instruction
            elif step == "execute_switch_operation":
                tool_kwargs["instruction"] = switch_instruction
                tool_kwargs["resolution"] = switch_resolution
                tool_kwargs["run_loadflow_after"] = True
            elif step == "summarize_switch_result":
                tool_kwargs["result_payload"] = switch_execution
                tool_kwargs["user_input"] = user_input
            elif step == "interpret_data_query_instruction":
                tool_kwargs["user_input"] = user_input
                tool_kwargs["inventory"] = data_inventory_result["inventory"] if isinstance(data_inventory_result, dict) else {}
            elif step == "resolve_pf_object_from_inventory_llm":
                tool_kwargs["instruction"] = data_instruction
                tool_kwargs["inventory"] = data_inventory_result["inventory"] if isinstance(data_inventory_result, dict) else {}
            elif step == "list_available_object_attributes":
                tool_kwargs["instruction"] = data_instruction
                tool_kwargs["resolution"] = data_resolution
            elif step == "select_pf_object_attributes_llm":
                tool_kwargs["instruction"] = data_instruction
                tool_kwargs["resolution"] = data_resolution
                tool_kwargs["attribute_listing"] = data_attribute_listing
            elif step == "read_pf_object_attributes":
                tool_kwargs["instruction"] = (data_attribute_selection.get("instruction") if isinstance(data_attribute_selection, dict) else None) or data_instruction
                tool_kwargs["resolution"] = data_resolution

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
            elif step == "interpret_switch_instruction":
                switch_instruction = result["instruction"]
            elif step == "resolve_switch_from_inventory_llm":
                switch_resolution = result
            elif step == "execute_switch_operation":
                switch_execution = result
            elif step == "summarize_switch_result":
                switch_summary = result
                summary = result
            elif step == "build_data_inventory":
                data_inventory_result = result
            elif step == "interpret_data_query_instruction":
                data_instruction = result["instruction"]
            elif step == "resolve_pf_object_from_inventory_llm":
                data_resolution = result
            elif step == "list_available_object_attributes":
                data_attribute_listing = result
            elif step == "select_pf_object_attributes_llm":
                data_attribute_selection = result
                data_instruction = result.get("instruction", data_instruction)
            elif step == "read_pf_object_attributes":
                data_query_result = result

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
            switch_instruction=switch_instruction,
            switch_resolution=switch_resolution,
            switch_execution=switch_execution,
            switch_summary=switch_summary,
            data_inventory_result=data_inventory_result,
            data_instruction=data_instruction,
            data_resolution=data_resolution,
            data_attribute_listing=data_attribute_listing,
            data_attribute_selection=data_attribute_selection,
            data_query_result=data_query_result,
            data_summary=data_summary,
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
        switch_instruction: Dict[str, Any] | None,
        switch_resolution: Dict[str, Any] | None,
        switch_execution: Dict[str, Any] | None,
        switch_summary: Dict[str, Any] | None,
        data_inventory_result: Dict[str, Any] | None,
        data_instruction: Dict[str, Any] | None,
        data_resolution: Dict[str, Any] | None,
        data_attribute_listing: Dict[str, Any] | None,
        data_attribute_selection: Dict[str, Any] | None,
        data_query_result: Dict[str, Any] | None,
        data_summary: Dict[str, Any] | None,
        debug_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer = summary.get("answer", "") if isinstance(summary, dict) else ""

        return {
            "status": "ok",
            "tool": "powerfactory",
            "agent": "PowerFactoryDomainAgent",
            "project": services.get("project_name"),
            "studycase": execution.get("studycase") if isinstance(execution, dict) else (switch_execution.get("studycase") if isinstance(switch_execution, dict) else (data_query_result.get("studycase") if isinstance(data_query_result, dict) else None)),
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
            "switch": {
                "instruction": switch_instruction,
                "resolution": switch_resolution,
                "execution": switch_execution,
                "summary": switch_summary,
            },
            "data_query": {
                "inventory": data_inventory_result.get("inventory", {}) if isinstance(data_inventory_result, dict) else {},
                "instruction": data_instruction,
                "resolution": data_resolution,
                "attribute_listing": data_attribute_listing,
                "attribute_selection": data_attribute_selection,
                "execution": data_query_result,
                "summary": data_summary,
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
                "switch_instruction": switch_instruction,
                "switch_resolution": switch_resolution,
                "switch_execution": switch_execution,
                "switch_summary": switch_summary,
                "data_inventory_result": data_inventory_result,
                "data_instruction": data_instruction,
                "data_resolution": data_resolution,
                "data_attribute_listing": data_attribute_listing,
                "data_attribute_selection": data_attribute_selection,
                "data_query_result": data_query_result,
                "data_summary": data_summary,
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