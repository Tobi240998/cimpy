from __future__ import annotations

from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# DEBUG HELPERS
# ------------------------------------------------------------------

def simplify_pf_debug_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, list):
        if len(value) > 10:
            return {
                "type": "list",
                "length": len(value),
                "preview": [simplify_pf_debug_value(item) for item in value[:5]],
            }
        return [simplify_pf_debug_value(item) for item in value]

    if isinstance(value, dict):
        simplified: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "services",
                "app",
                "project",
                "studycase",
                "study_case_obj",
                "project_obj",
                "pf",
                "topology_graph",
            }:
                simplified[key] = "<omitted>"
            else:
                simplified[key] = simplify_pf_debug_value(item)
        return simplified

    return repr(value)


def build_pf_tool_kwargs_debug(tool_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    debug_kwargs: Dict[str, Any] = {}

    for key, value in tool_kwargs.items():
        if key == "services":
            if isinstance(value, dict):
                debug_kwargs[key] = {
                    "status": value.get("status"),
                    "project_name": value.get("project_name"),
                    "studycase": getattr(value.get("studycase"), "loc_name", None),
                }
            else:
                debug_kwargs[key] = "<services>"
        else:
            debug_kwargs[key] = simplify_pf_debug_value(value)

    return debug_kwargs


# ------------------------------------------------------------------
# STATE INIT
# ------------------------------------------------------------------

def init_pf_execution_state() -> Dict[str, Any]:
    return {
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
        "delegated_result_subrequest": None,
    }


# ------------------------------------------------------------------
# TOOL KWARGS
# ------------------------------------------------------------------

def _safe_inventory(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        inventory = result.get("inventory", {})
        if isinstance(inventory, dict):
            return inventory
    return {}


def _safe_topology_graph(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("topology_graph")
    return None


def build_pf_tool_kwargs(
    step: str,
    services: Dict[str, Any],
    effective_user_input: str,
    classification: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    tool_kwargs: Dict[str, Any] = {"services": services}

    if step == "get_load_catalog":
        tool_kwargs["user_input"] = effective_user_input

    elif step == "summarize_load_catalog":
        tool_kwargs["catalog_result"] = state.get("catalog_result")
        tool_kwargs["user_input"] = effective_user_input

    elif step == "interpret_instruction":
        tool_kwargs["user_input"] = effective_user_input

    elif step == "resolve_load":
        tool_kwargs["instruction"] = state.get("instruction")

    elif step == "execute_change_load":
        tool_kwargs["instruction"] = state.get("instruction")

    elif step == "summarize_powerfactory_result":
        tool_kwargs["result_payload"] = state.get("execution")
        tool_kwargs["user_input"] = effective_user_input

    elif step == "build_topology_graph":
        tool_kwargs["contract_cubicles"] = True

    elif step == "build_topology_inventory":
        tool_kwargs["topology_graph_result"] = state.get("graph_result")

    elif step == "interpret_entity_instruction":
        tool_kwargs["user_input"] = effective_user_input
        tool_kwargs["inventory"] = _safe_inventory(state.get("inventory_result"))

    elif step == "resolve_entity_from_inventory":
        tool_kwargs["instruction"] = state.get("entity_instruction")
        tool_kwargs["inventory"] = _safe_inventory(state.get("inventory_result"))
        tool_kwargs["topology_graph"] = _safe_topology_graph(state.get("graph_result"))
        tool_kwargs["max_matches"] = 10

    elif step == "query_topology_neighbors":
        entity_resolution = state.get("entity_resolution") or {}
        selected_match = entity_resolution.get("selected_match", {}) if isinstance(entity_resolution, dict) else {}

        tool_kwargs["topology_graph"] = _safe_topology_graph(state.get("graph_result"))
        tool_kwargs["asset_query"] = (
            entity_resolution.get("asset_query")
            if isinstance(entity_resolution, dict)
            else effective_user_input
        )
        tool_kwargs["selected_node_id"] = (
            selected_match.get("node_id")
            if isinstance(selected_match, dict)
            else None
        )
        tool_kwargs["matches"] = (
            entity_resolution.get("matches", [])
            if isinstance(entity_resolution, dict)
            else []
        )
        tool_kwargs["max_matches"] = 10

    elif step == "summarize_topology_result":
        tool_kwargs["topology_result"] = state.get("topology_result")
        tool_kwargs["graph_result"] = state.get("graph_result")
        tool_kwargs["inventory_result"] = state.get("inventory_result")
        tool_kwargs["entity_instruction"] = state.get("entity_instruction")
        tool_kwargs["entity_resolution"] = state.get("entity_resolution")

    elif step == "interpret_switch_instruction":
        tool_kwargs["user_input"] = effective_user_input
        tool_kwargs["inventory"] = _safe_inventory(state.get("unified_inventory_result"))

    elif step == "build_unified_inventory":
        if classification.get("intent") == "change_switch_state":
            tool_kwargs["allowed_types"] = ["switch"]
        else:
            tool_kwargs["allowed_types"] = [
                "bus",
                "load",
                "line",
                "transformer",
                "generator",
                "switch",
            ]

    elif step == "resolve_objects_from_inventory_llm":
        if classification.get("intent") == "change_switch_state":
            tool_kwargs["instruction"] = state.get("switch_instruction")
        else:
            tool_kwargs["instruction"] = state.get("data_query_instruction")

        tool_kwargs["inventory"] = _safe_inventory(state.get("unified_inventory_result"))

    elif step == "execute_switch_operation":
        tool_kwargs["instruction"] = state.get("switch_instruction")
        tool_kwargs["resolution"] = state.get("object_resolution")
        tool_kwargs["run_loadflow_after"] = True

    elif step == "summarize_switch_result":
        tool_kwargs["result_payload"] = state.get("switch_execution")
        tool_kwargs["user_input"] = effective_user_input

    elif step == "interpret_data_query_instruction":
        tool_kwargs["user_input"] = effective_user_input
        tool_kwargs["inventory"] = _safe_inventory(state.get("unified_inventory_result"))

    elif step == "classify_data_source":
        tool_kwargs["instruction"] = state.get("data_query_instruction")

    elif step == "list_available_object_attributes":
        tool_kwargs["instruction"] = state.get("data_query_instruction")
        tool_kwargs["resolution"] = state.get("object_resolution")

    elif step == "select_pf_object_attributes_llm":
        tool_kwargs["instruction"] = state.get("data_query_instruction")
        tool_kwargs["resolution"] = state.get("object_resolution")
        tool_kwargs["attribute_listing"] = state.get("data_attribute_listing")

    elif step == "read_pf_object_attributes":
        selected_instruction = None
        data_attribute_selection = state.get("data_attribute_selection")
        if isinstance(data_attribute_selection, dict):
            selected_instruction = data_attribute_selection.get("instruction")

        tool_kwargs["instruction"] = selected_instruction or state.get("data_query_instruction")
        tool_kwargs["resolution"] = state.get("object_resolution")

    elif step == "summarize_pf_object_data_result":
        tool_kwargs["result_payload"] = state.get("data_query_execution")
        tool_kwargs["user_input"] = effective_user_input

    elif step == "unsupported_request":
        tool_kwargs["user_input"] = effective_user_input
        tool_kwargs["classification"] = classification

    return tool_kwargs


# ------------------------------------------------------------------
# STATE UPDATE
# ------------------------------------------------------------------

def store_pf_step_result(
    step: str,
    result: Dict[str, Any],
    state: Dict[str, Any],
) -> None:
    if step == "get_load_catalog":
        state["catalog_result"] = result

    elif step == "summarize_load_catalog":
        state["summary"] = result

    elif step == "interpret_instruction":
        state["instruction"] = result.get("instruction")

    elif step == "resolve_load":
        state["resolution"] = result

    elif step == "execute_change_load":
        state["execution"] = result

    elif step == "summarize_powerfactory_result":
        state["summary"] = result

    elif step == "build_topology_graph":
        state["graph_result"] = result

    elif step == "build_topology_inventory":
        state["inventory_result"] = result

    elif step == "interpret_entity_instruction":
        state["entity_instruction"] = result.get("instruction")

    elif step == "resolve_entity_from_inventory":
        state["entity_resolution"] = result

    elif step == "query_topology_neighbors":
        state["topology_result"] = result

    elif step == "summarize_topology_result":
        state["summary"] = result

    elif step == "interpret_switch_instruction":
        state["switch_instruction"] = result.get("instruction")

    elif step == "build_unified_inventory":
        state["unified_inventory_result"] = result

    elif step == "resolve_objects_from_inventory_llm":
        state["object_resolution"] = result

    elif step == "execute_switch_operation":
        state["switch_execution"] = result

    elif step == "summarize_switch_result":
        state["switch_summary"] = result
        state["summary"] = result

    elif step == "interpret_data_query_instruction":
        state["data_query_instruction"] = result.get("instruction")

    elif step == "classify_data_source":
        state["data_query_instruction"] = result.get("instruction")
        state["data_source_decision"] = result

    elif step == "list_available_object_attributes":
        state["data_attribute_listing"] = result

    elif step == "select_pf_object_attributes_llm":
        state["data_attribute_selection"] = result

    elif step == "read_pf_object_attributes":
        state["data_query_execution"] = result

    elif step == "summarize_pf_object_data_result":
        state["data_query_summary"] = result
        state["summary"] = result


# ------------------------------------------------------------------
# SUMMARY HELPERS
# ------------------------------------------------------------------

def merge_pf_summary_results(
    plan: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    summary_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if summary_results:
        messages: List[str] = []

        for item in summary_results:
            if not isinstance(item, dict):
                continue

            answer = item.get("answer")
            if isinstance(answer, str) and answer.strip():
                messages.append(answer.strip())

            item_messages = item.get("messages", [])
            if isinstance(item_messages, list):
                for message in item_messages:
                    if isinstance(message, str) and message.strip() and message.strip() not in messages:
                        messages.append(message.strip())

        if messages:
            return {
                "status": "ok",
                "tool": "merged_powerfactory_summary",
                "answer": "\n".join(messages),
                "messages": messages,
                "summary_count": len(summary_results),
                "plan_steps": [
                    item.get("step")
                    for item in plan
                    if isinstance(item, dict)
                ],
            }

    if isinstance(summary, dict):
        return summary

    return {
        "status": "ok",
        "tool": "empty_powerfactory_summary",
        "answer": "",
        "messages": [],
    }


def build_pf_generic_summary(
    user_input: str,
    plan: List[Dict[str, Any]],
    catalog_result: Optional[Dict[str, Any]],
    topology_result: Optional[Dict[str, Any]],
    entity_resolution: Optional[Dict[str, Any]],
    switch_execution: Optional[Dict[str, Any]],
    object_resolution: Optional[Dict[str, Any]],
    data_attribute_listing: Optional[Dict[str, Any]],
    data_query_execution: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    plan_steps = [
        item.get("step")
        for item in plan
        if isinstance(item, dict)
    ]

    if isinstance(catalog_result, dict) and catalog_result.get("status") == "ok":
        entries = catalog_result.get("loads", []) or []
        requested_type = catalog_result.get("requested_type", "object")
        count = catalog_result.get("count", len(entries))

        names = [
            item.get("loc_name") or item.get("name") or item.get("full_name")
            for item in entries
            if isinstance(item, dict)
        ]
        names = [name for name in names if name]

        if names:
            preview = ", ".join(str(name) for name in names[:20])
            if len(names) > 20:
                preview += f", ... ({len(names)} gesamt)"
            answer = f"Im aktiven PowerFactory-Projekt wurden {count} Objekte vom Typ '{requested_type}' gefunden: {preview}"
        else:
            answer = f"Im aktiven PowerFactory-Projekt wurden {count} Objekte vom Typ '{requested_type}' gefunden."

        return {
            "status": "ok",
            "tool": "summarize_generic_result",
            "answer": answer,
            "messages": [answer],
            "source": "catalog_result",
            "plan_steps": plan_steps,
        }

    if isinstance(topology_result, dict) and topology_result.get("status") == "ok":
        selected_node = topology_result.get("selected_node") or {}
        neighbors = topology_result.get("neighbors", []) or []
        neighbor_count = topology_result.get("neighbor_count", len(neighbors))

        selected_name = (
            selected_node.get("name")
            or selected_node.get("full_name")
            or selected_node.get("node_id")
            or "das ausgewählte Objekt"
        )

        if neighbor_count == 0:
            answer = f"Für '{selected_name}' wurden keine direkten Nachbarn im PowerFactory-Topologiegraphen gefunden."
        else:
            preview_items = []
            for neighbor in neighbors[:10]:
                if not isinstance(neighbor, dict):
                    continue
                neighbor_name = neighbor.get("name") or neighbor.get("full_name") or "<unbekannt>"
                neighbor_class = neighbor.get("pf_class") or "<unknown>"
                preview_items.append(f"{neighbor_name} ({neighbor_class})")

            if preview_items:
                answer = (
                    f"Für '{selected_name}' wurden {neighbor_count} direkte Nachbarn gefunden: "
                    + ", ".join(preview_items)
                )
            else:
                answer = f"Für '{selected_name}' wurden {neighbor_count} direkte Nachbarn gefunden."

        return {
            "status": "ok",
            "tool": "summarize_generic_result",
            "answer": answer,
            "messages": [answer],
            "source": "topology_result",
            "plan_steps": plan_steps,
        }

    if isinstance(switch_execution, dict) and switch_execution.get("status") == "ok":
        switch_info = switch_execution.get("switch", {}) if isinstance(switch_execution.get("switch"), dict) else {}
        switch_name = switch_info.get("name") or switch_info.get("full_name") or "der Schalter"
        state_before = switch_execution.get("state_before")
        state_after = switch_execution.get("state_after")

        answer = f"Die Schalthandlung für '{switch_name}' wurde ausgeführt."
        if state_before is not None or state_after is not None:
            answer += f" Zustand vorher: {state_before}, Zustand nachher: {state_after}."

        return {
            "status": "ok",
            "tool": "summarize_generic_result",
            "answer": answer,
            "messages": [answer],
            "source": "switch_execution",
            "plan_steps": plan_steps,
        }

    if isinstance(data_query_execution, dict) and data_query_execution.get("status") == "ok":
        answer = data_query_execution.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {
                "status": "ok",
                "tool": "summarize_generic_result",
                "answer": answer,
                "messages": [answer],
                "source": "data_query_execution",
                "plan_steps": plan_steps,
            }

    if isinstance(data_attribute_listing, dict) and data_attribute_listing.get("status") == "ok":
        options = data_attribute_listing.get("attribute_options", []) or []
        object_info = data_attribute_listing.get("object", {}) if isinstance(data_attribute_listing.get("object"), dict) else {}
        object_name = object_info.get("name") or object_info.get("full_name") or "das Objekt"

        labels = []
        for option in options[:30]:
            if not isinstance(option, dict):
                continue
            label = (
                option.get("label")
                or option.get("attribute_name")
                or option.get("field_name")
                or option.get("handle")
            )
            if label:
                labels.append(str(label))

        if labels:
            answer = f"Für '{object_name}' sind unter anderem folgende Attribute verfügbar: " + ", ".join(labels)
        else:
            answer = f"Für '{object_name}' wurden keine gut darstellbaren Attribute gefunden."

        return {
            "status": "ok",
            "tool": "summarize_generic_result",
            "answer": answer,
            "messages": [answer],
            "source": "data_attribute_listing",
            "plan_steps": plan_steps,
        }

    if isinstance(object_resolution, dict) and object_resolution.get("status") == "ok":
        selected = object_resolution.get("selected_match") or {}
        selected_name = selected.get("name") or selected.get("full_name") or object_resolution.get("asset_query")

        if selected_name:
            answer = f"Das PowerFactory-Objekt wurde aufgelöst: {selected_name}."
            return {
                "status": "ok",
                "tool": "summarize_generic_result",
                "answer": answer,
                "messages": [answer],
                "source": "object_resolution",
                "plan_steps": plan_steps,
            }

    return None


# ------------------------------------------------------------------
# FINAL RESULT BUILDERS
# ------------------------------------------------------------------

def build_pf_success_result(
    services: Dict[str, Any],
    user_input: str,
    classification: Dict[str, Any],
    plan: List[Dict[str, Any]],
    instruction: Optional[Dict[str, Any]],
    resolution: Optional[Dict[str, Any]],
    execution: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    summary_results: List[Dict[str, Any]],
    catalog_result: Optional[Dict[str, Any]],
    graph_result: Optional[Dict[str, Any]],
    inventory_result: Optional[Dict[str, Any]],
    entity_instruction: Optional[Dict[str, Any]],
    entity_resolution: Optional[Dict[str, Any]],
    topology_result: Optional[Dict[str, Any]],
    switch_instruction: Optional[Dict[str, Any]],
    switch_execution: Optional[Dict[str, Any]],
    switch_summary: Optional[Dict[str, Any]],
    data_query_instruction: Optional[Dict[str, Any]],
    object_resolution: Optional[Dict[str, Any]],
    data_attribute_listing: Optional[Dict[str, Any]],
    data_attribute_selection: Optional[Dict[str, Any]],
    data_query_execution: Optional[Dict[str, Any]],
    data_query_summary: Optional[Dict[str, Any]],
    debug_trace: List[Dict[str, Any]],
    available_tools: List[Any],
    planning_debug: Optional[Dict[str, Any]] = None,
    debug_mode: bool = True,
) -> Dict[str, Any]:
    final_summary = merge_pf_summary_results(
        plan=plan,
        summary=summary,
        summary_results=summary_results,
    )

    answer = final_summary.get("answer", "") if isinstance(final_summary, dict) else ""
    messages = final_summary.get("messages", []) if isinstance(final_summary, dict) else []

    if not isinstance(final_summary, dict) or not str(answer).strip():
        generic_summary = build_pf_generic_summary(
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
            answer = final_summary.get("answer", "")
            messages = final_summary.get("messages", [])

    result = {
        "status": "ok",
        "tool": "powerfactory",
        "agent": "SingleAgent",
        "project": services.get("project_name"),
        "studycase": (
            execution.get("studycase")
            if isinstance(execution, dict)
            else (
                switch_execution.get("studycase")
                if isinstance(switch_execution, dict)
                else (
                    data_query_execution.get("studycase")
                    if isinstance(data_query_execution, dict)
                    else None
                )
            )
        ),
        "user_input": user_input,
        "classification": classification,
        "plan": plan,
        "available_tools": available_tools,
        "instruction": instruction,
        "resolved_load": execution.get("resolved_load") if isinstance(execution, dict) else None,
        "data": (
            execution.get("data", {})
            if isinstance(execution, dict)
            else (
                data_query_execution.get("data", {})
                if isinstance(data_query_execution, dict)
                else {}
            )
        ),
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

    if debug_mode:
        result["debug"] = {
            "planning": planning_debug or {},
            "selected_equipment": {
                "resolved_load": execution.get("resolved_load") if isinstance(execution, dict) else None,
                "entity_resolution": entity_resolution,
                "object_resolution": object_resolution,
            },
            "trace": debug_trace,
        }

    return result


def build_pf_error_result(
    error_result: Dict[str, Any],
    debug_trace: List[Dict[str, Any]],
    available_tools: List[Any],
    planning_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = dict(error_result)
    result["agent"] = "SingleAgent"
    result["available_tools"] = available_tools

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        error_code = result.get("error", "unknown_error")
        details = result.get("details", "")
        if isinstance(details, str) and details.strip():
            result["answer"] = (
                f"Die Anfrage konnte nicht sauber ausgeführt werden ({error_code}). "
                f"Details: {details}"
            )
        else:
            result["answer"] = f"Die Anfrage konnte nicht sauber ausgeführt werden ({error_code})."

    result["debug"] = {
        "planning": planning_debug or {},
        "trace": debug_trace,
    }

    return result