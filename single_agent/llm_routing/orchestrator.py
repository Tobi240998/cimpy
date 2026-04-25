from typing import Any, Dict

from cimpy.single_agent.llm_routing.single_domain_agent import SingleDomainAgent


class Orchestrator:
    def __init__(self):
        self.agent = SingleDomainAgent()

    def handle(self, user_input: str) -> Dict[str, Any]:
        return self.agent.run(user_input)