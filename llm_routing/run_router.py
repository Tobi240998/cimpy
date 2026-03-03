from cimpy.llm_routing.orchestrator import Orchestrator

if __name__ == "__main__":
    orch = Orchestrator()

    print("Router gestartet. Tippe 'exit' zum Beenden.\n")
    while True:
        text = input(">>> ")
        if text.strip().lower() in ("exit", "quit"):
            break

        out = orch.handle(text)

        if out.get("route") == "ASK_USER":
            print(out["question"])
        else:
            res = out.get("result", {})
            if isinstance(res, dict) and "answer" in res:
                print(res["answer"])
            else:
                print(out)

        print()