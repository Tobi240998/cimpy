from test_multiagenten.agents.llm_routing.orchestrator import Orchestrator

def historic_executor(user_input: str):
    return {"note": "CIM Historie (TODO)", "input": user_input}

def pf_executor(user_input: str):
    return {"note": "PowerFactory (TODO)", "input": user_input}

if __name__ == "__main__":
    orch = Orchestrator(historic_executor=historic_executor, pf_executor=pf_executor)

    print("Router gestartet. Tippe 'exit' zum Beenden.\n")
    while True:
        text = input(">>> ")
        if text.strip().lower() in ("exit", "quit"):
            break
        print(orch.handle(text))
        print()