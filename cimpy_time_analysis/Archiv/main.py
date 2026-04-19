# cim_historical/cli.py

from cimpy_time_analysis.runner import run_historical_cim_analysis


if __name__ == "__main__":
    cim_root = r"C:\Users\STELLER\Documents\Masterarbeit\CIM-Dateien\tobias_CIM_daten\data\extracted"

    user_input = "Wie verhält sich der Trafo 19 - 20 über den Tag?"

    result = run_historical_cim_analysis(
        user_input=user_input,
        cim_root=cim_root
    )

    print("\nAntwort:\n")
    print(result["answer"])