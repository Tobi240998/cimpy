from cimpy.llm_routing.LLM_routeAgent import LLM_routeAgent


class Orchestrator:

    def __init__(self, historic_executor, pf_executor):
        self.router = LLM_routeAgent()
        self.historic_executor = historic_executor
        self.pf_executor = pf_executor

    def handle(self, user_input: str):

        decision = self.router.route(user_input)
        route = decision.get("route", "HISTORIC")

        if route == "POWERFACTORY":
            result = self.pf_executor(user_input)
        else:
            result = self.historic_executor(user_input)

        return {
            "route": route,
            "result": result
        }