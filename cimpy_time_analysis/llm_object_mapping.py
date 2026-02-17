LLM_OBJECT_MAP = {
    "trafo": "PowerTransformer",
    "transformator": "PowerTransformer",
    "trf": "PowerTransformer",
    "leistung": "SvPowerFlow",
    "spannung": "SvVoltage",
    "bus": "BusbarSection",
    "knoten": "BusbarSection",
    "last": "EnergyConsumer",
    "verbraucher": "EnergyConsumer"
}


def interpret_user_query(user_input: str):
    """
    Erkennt nur die CIM-Objekttypen (Intent/Scope).
    Konkrete Objekte (z.B. welcher Trafo) werden im Backend deterministisch gematcht.
    """
    user_input = user_input.lower()
    detected = set()

    for keyword, cim_type in LLM_OBJECT_MAP.items():
        if keyword in user_input:
            detected.add(cim_type)

    return list(detected)
