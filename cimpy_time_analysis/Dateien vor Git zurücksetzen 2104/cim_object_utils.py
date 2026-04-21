def collect_all_cim_objects(container):
    """
    Rekursives Einsammeln aller echten CIMpy-Objekte
    (inkl. SvVoltage, SvPowerFlow etc.)
    Ablauf: Prüfen, ob Objekt ein dict oder eine list ist -> eine Ebene weiter unten von vorne starten -> wenn kein dict oder list -> prüfen, ob aus CIMpy, falls ja, in Objects anhängen 
    """
    objects = []

    if isinstance(container, dict):
        for value in container.values():
            objects.extend(collect_all_cim_objects(value))

    elif isinstance(container, list):
        for value in container:
            objects.extend(collect_all_cim_objects(value))

    else:
        cls = container.__class__
        if cls.__module__.startswith("cimpy"):
            objects.append(container)

    return objects
