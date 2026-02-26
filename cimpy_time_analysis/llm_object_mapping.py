LLM_OBJECT_MAP = {
    # Equipment
    "trafo": "PowerTransformer",
    "transformator": "PowerTransformer",
    "transformer": "PowerTransformer",

    # Lasten/Verbraucher 
    "verbraucher": "ConformLoad",
    "last": "ConformLoad",
    "load": "ConformLoad",
    "conformload": "ConformLoad",

    # State variables / queries
    "spannung": "SvVoltage",
    "voltage": "SvVoltage",

    # Leistung: 
    "leistung": "SvPowerFlow",
    "power": "SvPowerFlow",

    # Metrik-Hinweise
    "wirkleistung": "METRIC_P",
    "p": "METRIC_P",

    "blindleistung": "METRIC_Q",
    "q": "METRIC_Q",

    "scheinleistung": "METRIC_S",
    "s": "METRIC_S",
}


def interpret_user_query(user_input: str):
    
    user_input_l = user_input.lower()
    detected = set()
    metric = None

    for keyword, cim_type in LLM_OBJECT_MAP.items():
        if keyword in user_input_l:
            if cim_type.startswith("METRIC_"):
                metric = cim_type.split("_", 1)[1]  # "P"/"Q"/"S"
            else:
                detected.add(cim_type)

    return {
        "detected_types": list(detected),
        "metric": metric
    }