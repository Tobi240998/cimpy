from cimpy.llm_routing.orchestrator import Orchestrator
from cimpy.llm_routing.registry import historic_executor, pf_executor

if __name__ == "__main__":
    orch = Orchestrator(historic_executor=historic_executor, pf_executor=pf_executor)

    print("Router gestartet. Tippe 'exit' zum Beenden.\n")
    while True:
        text = input(">>> ")
        if text.strip().lower() in ("exit", "quit"):
            break
        out = orch.handle(text)
        print(out["answer"] if isinstance(out, dict) and "answer" in out else out)
        print()