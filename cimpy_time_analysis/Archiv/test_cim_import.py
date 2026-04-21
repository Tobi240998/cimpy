from cimpy.cimpy_time_analysis.cim_domain_agent import CIMDomainAgent
from cimpy.llm_routing.config import CIM_ROOT

agent = CIMDomainAgent(cim_root=CIM_ROOT)

result = agent.run("Zeige mir die Nachbarn von Trafo 19 - 20 am 2026-01-09")
print(result)