import sys
import traceback

# === ANPASSEN ===
PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP7\Python\3.10"
PROJECT_NAME = r"Nine-bus System(2)"  # optional: später exakt aus Liste übernehmen
# =================


def _to_py_list(obj):
    """
    Macht aus typischen PowerFactory Rückgaben eine echte Python-Liste.
    GetContents(...) liefert je nach Version/Objekt u.a.:
      - list
      - [list] (wrapped)
      - DataObject / Collection-ähnlich (kein len)
    """
    if obj is None:
        return []

    # Fall 1: bereits eine Python-Liste
    if isinstance(obj, list):
        # häufig: [ [objs...] ] oder [objs...]
        if len(obj) == 1 and isinstance(obj[0], list):
            return obj[0]
        return obj

    # Fall 2: PF-Collections manchmal iterierbar
    try:
        return list(obj)
    except Exception:
        pass

    # Fall 3: manche DataObjects haben GetChildren()
    try:
        children = obj.GetChildren()
        if children is None:
            return []
        if isinstance(children, list):
            return children
        try:
            return list(children)
        except Exception:
            return []
    except Exception:
        pass

    # Unbekannt -> leere Liste
    return []


def main():
    print("== PowerFactory Python Selbsttest ==")

    if PF_PYTHON_PATH not in sys.path:
        sys.path.append(PF_PYTHON_PATH)

    try:
        import powerfactory as pf  # type: ignore
    except Exception as e:
        print("FEHLER: powerfactory Import fehlgeschlagen")
        print(e)
        return

    print("OK: powerfactory Modul importiert")

    # App holen
    app = None
    try:
        if hasattr(pf, "GetApplicationExt"):
            app = pf.GetApplicationExt()
            print("Info: pf.GetApplicationExt() verwendet")
        else:
            app = pf.GetApplication()
            print("Info: pf.GetApplication() verwendet")
    except Exception:
        print("FEHLER: GetApplication/GetApplicationExt Exception")
        traceback.print_exc()
        return

    if app is None:
        print("FEHLER: Keine PowerFactory Application (app is None)")
        return

    try:
        app.Show()
    except Exception:
        pass

    print("OK: PowerFactory Anwendung verbunden:", app)

    # Current User
    try:
        user = app.GetCurrentUser()
    except Exception:
        print("FEHLER: GetCurrentUser() Exception")
        traceback.print_exc()
        return

    if user is None:
        print("FEHLER: GetCurrentUser() liefert None (kein User-Kontext)")
        return

    print("OK: Current User:", getattr(user, "loc_name", str(user)))

    # Projekte auflisten
    print("\n--- Projekte unter aktuellem User (*.IntPrj) ---")
    try:
        raw = user.GetContents("*.IntPrj", 0)
    except Exception:
        print("FEHLER: user.GetContents('*.IntPrj', 0) Exception")
        traceback.print_exc()
        return

    projects = _to_py_list(raw)

    if not projects:
        print("Keine Projekte gefunden (oder Zugriff fehlt).")
        print("Debug: type(raw) =", type(raw))
        print("Debug: raw =", raw)
        return

    print(f"{len(projects)} Projekte gefunden:")
    for p in projects:
        print(" -", getattr(p, "loc_name", str(p)))

    # Optional: Projekt aktivieren
    print(f"\n--- Aktivierungstest: '{PROJECT_NAME}' ---")
    target = None
    for p in projects:
        if getattr(p, "loc_name", None) == PROJECT_NAME:
            target = p
            break

    if target is None:
        print("Hinweis: PROJECT_NAME nicht exakt gefunden.")
        print("Tipp: Kopiere den Namen 1:1 aus der Liste oben in PROJECT_NAME.")
        print("Aktives Projekt wird daher NICHT verändert.")
        return

    try:
        target.Activate()
        print("OK: target.Activate() ausgeführt")
    except Exception:
        print("FEHLER: target.Activate() Exception")
        traceback.print_exc()
        return

    # Prüfen: aktives Projekt
    try:
        active_project = app.GetActiveProject()
    except Exception:
        print("FEHLER: GetActiveProject() Exception")
        traceback.print_exc()
        return

    if active_project is None:
        print("FEHLER: Nach Activate() ist GetActiveProject() weiterhin None")
        print("Mögliche Ursachen: Lizenz/Workspace/User-Kontext passt nicht.")
        return

    print("OK: Aktives Projekt:", getattr(active_project, "loc_name", active_project))

    # Ab hier sollten Projektfolder funktionieren
    print("\n--- Projektordner-Test (study) ---")
    try:
        study_folder = app.GetProjectFolder("study")
        print("study folder:", study_folder)
    except Exception:
        print("Warnung: GetProjectFolder('study') hat eine Exception geworfen")
        traceback.print_exc()

    # Aktiver Study Case
    try:
        sc = app.GetActiveStudyCase()
    except Exception:
        sc = None
    print("Active Study Case:", getattr(sc, "loc_name", sc) if sc else None)

    print("\nSelbsttest abgeschlossen.")


if __name__ == "__main__":
    main()