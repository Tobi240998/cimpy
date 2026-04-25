from __future__ import annotations

from typing import Any, Dict, List

from cimpy.single_agent.llm_routing.unified_plan import UnifiedPlan
from cimpy.single_agent.pf.powerfactory_mcp_tools import build_powerfactory_services
from cimpy.single_agent.llm_routing.unified_tool_registry import UnifiedToolRegistry

class UnifiedExecutor:
    def __init__(self, cim_agent, powerfactory_agent):
        self.cim_agent = cim_agent
        self.powerfactory_agent = powerfactory_agent

        self.registry = UnifiedToolRegistry(
            cim_registry=cim_agent.registry,
            pf_registry=powerfactory_agent.registry,
        )

    def execute(self, plan: UnifiedPlan) -> Dict[str, Any]:
        if plan.domain == "cim":
            return self._execute_cim(plan)

        if plan.domain == "powerfactory":
            return self._execute_powerfactory(plan)

        return {
            "status": "error",
            "answer": f"Unbekannte Domain im Plan: {plan.domain}",
        }

    def _plan_to_raw_items(self, plan: UnifiedPlan) -> List[Dict[str, Any]]:
        items = []

        for step in plan.steps:
            item = {
                "step": step.tool,
                "description": getattr(step, "description", ""),
            }

            step_args = getattr(step, "args", {}) or {}

            user_input_override = step_args.get("user_input_override")
            source_subrequest = step_args.get("source_subrequest")

            if user_input_override:
                item["user_input_override"] = user_input_override

            if source_subrequest:
                item["source_subrequest"] = source_subrequest

            items.append(item)

        return items

    def _execute_cim(self, plan: UnifiedPlan) -> Dict[str, Any]:
        raw_plan = self._plan_to_raw_items(plan)
        classification = getattr(plan, "classification", {}) or {}

        state: Dict[str, Any] = {
            "user_input": plan.user_input,
            "classification": classification,
        }

        debug_trace = []

        for item in raw_plan:
            step = item["step"]
            full_tool_name = f"cim.{step}"

            tool_kwargs = {
                **state,
                "cim_root": self.cim_agent.cim_root,
                "user_input": item.get("user_input_override", plan.user_input),
            }

            tool_spec = self.registry.get_tool_spec(full_tool_name)
            result = self.registry.invoke(full_tool_name, **tool_kwargs)

            debug_trace.append({
                "step": step,
                "tool_spec": {
                    "name": getattr(tool_spec, "name", step) if tool_spec else step,
                    "description": getattr(tool_spec, "description", "") if tool_spec else "",
                    "capability_tags": getattr(tool_spec, "capability_tags", []) if tool_spec else [],
                    "mutating": getattr(tool_spec, "mutating", False) if tool_spec else False,
                },
                "result": result,
            })

            if not isinstance(result, dict):
                return {
                    "status": "error",
                    "answer": f"CIM-Tool {step} returned non-dict result.",
                    "debug_trace": debug_trace,
                }

            if result.get("status") == "error":
                result["debug_trace"] = debug_trace
                return result

            state.update(result)

        state["debug_trace"] = debug_trace
        return state

    def _execute_powerfactory(self, plan: UnifiedPlan) -> Dict[str, Any]:
        pf_agent = self.powerfactory_agent
        raw_plan = self._plan_to_raw_items(plan)

        services = build_powerfactory_services(project_name=pf_agent.project_name)
        if services.get("status") != "ok":
            return services

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

        for item in raw_plan:
            step = item["step"]
            effective_user_input = item.get("user_input_override", plan.user_input)

            tool_kwargs = pf_agent._build_tool_kwargs(
                step=step,
                services=services,
                effective_user_input=effective_user_input,
                classification=getattr(plan, "classification", {}) or {},
                state=state,
            )

            full_tool_name = f"pf.{step}"
            tool_spec = self.registry.get_tool_spec(full_tool_name)
            result = self.registry.invoke(full_tool_name, **tool_kwargs)

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
                "tool_kwargs": pf_agent._build_tool_kwargs_debug(tool_kwargs),
                "result": result,
            })

            if result.get("status") != "ok":
                return pf_agent.build_error_result(
                    error_result=result,
                    debug_trace=debug_trace,
                )

            pf_agent._store_step_result(
                step=step,
                result=result,
                state=state,
            )

            if step == "interpret_instruction":
                instruction = state.get("instruction") or {}
                result_mode = instruction.get("result_request_mode")
                result_query_text = str(instruction.get("result_query_text") or "").strip()

                if result_mode == "delegate_result_query" and result_query_text:
                    state["delegated_result_subrequest"] = {
                        "user_input": result_query_text,
                        "depends_on_previous": True,
                        "source": "delegated_result_query",
                    }

            if pf_agent._is_summary_step(step):
                state["summary_results"].append(result)

        return pf_agent.build_success_result(
            services=services,
            user_input=plan.user_input,
            classification=getattr(plan, "classification", {}) or {},
            plan=raw_plan,
            instruction=state.get("instruction"),
            resolution=state.get("resolution"),
            execution=state.get("execution"),
            summary=state.get("summary"),
            summary_results=state.get("summary_results", []),
            catalog_result=state.get("catalog_result"),
            graph_result=state.get("graph_result"),
            inventory_result=state.get("inventory_result"),
            entity_instruction=state.get("entity_instruction"),
            entity_resolution=state.get("entity_resolution"),
            topology_result=state.get("topology_result"),
            switch_instruction=state.get("switch_instruction"),
            switch_execution=state.get("switch_execution"),
            switch_summary=state.get("switch_summary"),
            data_query_instruction=state.get("data_query_instruction"),
            object_resolution=state.get("object_resolution"),
            data_attribute_listing=state.get("data_attribute_listing"),
            data_attribute_selection=state.get("data_attribute_selection"),
            data_query_execution=state.get("data_query_execution"),
            data_query_summary=state.get("data_query_summary"),
            debug_trace=debug_trace,
        )