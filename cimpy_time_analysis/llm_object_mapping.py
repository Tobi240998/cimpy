# llm_object_mapping.py

LLM_OBJECT_MAP = {
    "trafo": "PowerTransformer",
    "transformator": "PowerTransformer",
    "leistung": "SvPowerFlow",
    "spannung": "SvVoltage",
    "bus": "BusbarSection",
    "knoten": "BusbarSection",
    "last": "EnergyConsumer",
    "verbraucher": "EnergyConsumer"
}


def interpret_user_query(user_input: str):
    user_input = user_input.lower()
    detected = set()

    for keyword, cim_type in LLM_OBJECT_MAP.items():
        if keyword in user_input:
            detected.add(cim_type)

    return list(detected)
