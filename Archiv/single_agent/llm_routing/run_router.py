from cimpy.single_agent.llm_routing.orchestrator import Orchestrator

if __name__ == "__main__":
    orch = Orchestrator()

    print("Router gestartet. Tippe 'exit' zum Beenden.\n")
    while True:
        text = input(">>> ")
        if text.strip().lower() in ("exit", "quit"): # Abbruch, wenn exit getippt wird
            break

        out = orch.handle(text)
    

        # falls Ergebnis der Funktion "ASK_USER" -> Frage wird gestellt 
        if out.get("route") == "ASK_USER":
            print(out["question"])
        else:
            # falls Eregebnis der Funktion "result" -> Antwort wird ausgegeben
            res = out.get("result", {})
            if isinstance(res, dict) and "answer" in res:
                print(res["answer"])
            else:
                print(out)

        print()