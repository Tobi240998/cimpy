# cim_historical/runner.py

from cimpy.cimpy_time_analysis.llm_cim_orchestrator import handle_user_query
from cimpy.cimpy_time_analysis.llm_object_mapping import interpret_user_query
from cimpy.cimpy_time_analysis.llm_analysis_planner import plan_analysis_with_debug
from cimpy.cimpy_time_analysis.cim_snapshot_cache import (
    preprocess_snapshots,
    preprocess_snapshots_for_states,
    summarize_snapshot_cache,
)
from cimpy.cimpy_time_analysis.load_cim_data import (
    scan_snapshot_inventory,
    load_base_snapshot,
    build_network_index_from_snapshot,
    load_snapshots_for_time_window,
    load_cim_snapshots,
)


def _extract_required_state_types_from_plan(analysis_plan: dict) -> list[str]:
    """
    Holt die gewünschten State-Typen aus dem Analyseplan.
    """
    if not analysis_plan:
        return []

    state_types = analysis_plan.get("needs_state_types", []) or []
    out = []
    seen = set()

    for s in state_types:
        if not s:
            continue
        s = str(s).strip()
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out


def _extract_required_state_types(parsed_query: dict, analysis_plan: dict | None = None) -> list[str]:
    """
    Leitet aus Analyseplan und geparster Nutzerfrage ab, welche State-Typen benötigt werden.

    Priorität:
    1) Analyseplan
    2) parsed_query
    """
    plan_state_types = _extract_required_state_types_from_plan(analysis_plan)
    if plan_state_types:
        return plan_state_types

    if not parsed_query:
        return []

    state_detected = parsed_query.get("state_detected", []) or []

    state_types = []
    if "SvVoltage" in state_detected:
        state_types.append("SvVoltage")
    if "SvPowerFlow" in state_detected:
        state_types.append("SvPowerFlow")

    out = []
    seen = set()
    for s in state_types:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out


def _load_relevant_snapshots(
    cim_root: str,
    parsed_query: dict,
    *,
    snapshot_inventory=None,
    preloaded_snapshots=None,
):
    """
    Lädt nur die Snapshots, die im erkannten Zeitfenster liegen.

    Falls preloaded_snapshots übergeben wurden, verwenden wir diese direkt weiter.
    """
    if preloaded_snapshots is not None:
        return preloaded_snapshots

    time_start = parsed_query.get("time_start", None) if parsed_query else None
    time_end = parsed_query.get("time_end", None) if parsed_query else None

    return load_snapshots_for_time_window(
        root_folder=cim_root,
        start_time=time_start,
        end_time=time_end,
        snapshot_inventory=snapshot_inventory,
    )


def _build_request_snapshot_cache(
    cim_snapshots: dict,
    parsed_query: dict,
    analysis_plan: dict | None = None,
):
    """
    Baut den Snapshot-Cache nur für die tatsächlich benötigten State-Typen.

    Topologie-only:
    - kein State erforderlich
    - leerer Cache reicht, weil handle_user_query(...) für Topologiefragen
      auf den network_index / topology_graph zugreift.
    """
    required_state_types = _extract_required_state_types(
        parsed_query=parsed_query,
        analysis_plan=analysis_plan,
    )

    if not cim_snapshots:
        return {}, required_state_types

    if not required_state_types:
        return {}, required_state_types

    snapshot_cache = preprocess_snapshots_for_states(
        cim_snapshots=cim_snapshots,
        state_types=required_state_types,
    )
    return snapshot_cache, required_state_types


def _should_load_structure(analysis_plan: dict | None) -> bool:
    """
    Ob Struktur / Basisindex benötigt wird.
    """
    if not analysis_plan:
        return True
    return bool(analysis_plan.get("needs_structure", True))


def _should_load_states(analysis_plan: dict | None, parsed_query: dict | None) -> bool:
    """
    Ob historische State-Daten geladen werden müssen.
    """
    required_state_types = _extract_required_state_types(
        parsed_query=parsed_query or {},
        analysis_plan=analysis_plan,
    )
    return len(required_state_types) > 0


def run_historical_cim_analysis(
    user_input: str,
    cim_root: str,
    preloaded_snapshots=None,
    preloaded_inventory=None,
    preloaded_base_snapshot=None,
    preloaded_network_index=None,
):
    """
    Führt eine historische CIM-Analyse mit vorgeschaltetem LLM-Planer und selektivem Laden aus.

    Ablauf:
    1. Snapshot-Inventar scannen
    2. Basissnapshot für statischen Netzindex / Topologie laden
    3. Netzwerkindex bauen
    4. LLM-Analyseplan erzeugen
    5. Nutzerfrage präziser parsen
    6. Nur relevante Snapshots im Zeitfenster laden
    7. Nur benötigte State-Typen preprocessen
    8. Analyse ausführen

    Returns
    -------
    dict
        Standardisiertes Result-Objekt.
    """

    # ---------------------------------------------------------
    # 1) Inventory / Discovery
    # ---------------------------------------------------------
    if preloaded_inventory is None:
        snapshot_inventory = scan_snapshot_inventory(cim_root)
    else:
        snapshot_inventory = preloaded_inventory

    # ---------------------------------------------------------
    # 2) Basis-Snapshot + Netzwerkindex
    #    Der Planner soll verfügbare Equipment-Typen sehen können.
    # ---------------------------------------------------------
    if preloaded_network_index is not None:
        network_index = preloaded_network_index
        base_snapshot = preloaded_base_snapshot
    else:
        if preloaded_base_snapshot is None:
            base_snapshot = load_base_snapshot(
                root_folder=cim_root,
                snapshot_inventory=snapshot_inventory,
            )
        else:
            base_snapshot = preloaded_base_snapshot

        network_index = build_network_index_from_snapshot(base_snapshot)

    # Schutz: ohne Basisindex keine sinnvolle Analyse
    if not network_index or not network_index.get("equipment_name_index"):
        return {
            "status": "error",
            "tool": "historical_cim_analysis",
            "input": user_input,
            "answer": "Es konnte kein gültiger Netzwerkindex aus dem Basissnapshot aufgebaut werden.",
            "debug": {
                "num_inventory_snapshots": len(snapshot_inventory.get("snapshots", [])) if snapshot_inventory else 0,
                "index_source_snapshot": None,
                "analysis_plan": {},
                "required_state_types": [],
                "num_loaded_snapshots": 0,
                "snapshot_cache_summary": {},
            },
        }

    # ---------------------------------------------------------
    # 3) LLM-Analyseplan erzeugen
    # ---------------------------------------------------------
    planner_result = plan_analysis_with_debug(
        user_input=user_input,
        snapshot_inventory=snapshot_inventory,
        network_index=network_index,
    )

    analysis_plan = planner_result.get("plan", {}) if planner_result else {}
    planner_context = planner_result.get("context", {}) if planner_result else {}

    # ---------------------------------------------------------
    # 4) Nutzerfrage präziser parsen
    #    -> Equipment-Auswahl, exakte Zeitfenster, State-Erkennung
    # ---------------------------------------------------------
    parsed_query = interpret_user_query(
    user_input=user_input,
    network_index=network_index,
    allowed_equipment_types=analysis_plan.get("target_equipment_types") or None,
    allowed_state_types=analysis_plan.get("needs_state_types") or None,
    require_time_window=bool(analysis_plan.get("requires_time_window", True)),
)

    # ---------------------------------------------------------
    # 5) Relevante Snapshots laden
    #    Nur wenn laut Plan / Parsing überhaupt State-Daten nötig sind.
    # ---------------------------------------------------------
    cim_snapshots = {}

    if _should_load_states(analysis_plan, parsed_query):
        cim_snapshots = _load_relevant_snapshots(
            cim_root=cim_root,
            parsed_query=parsed_query,
            snapshot_inventory=snapshot_inventory,
            preloaded_snapshots=preloaded_snapshots,
        )

        # Fallback:
        # Falls wegen fehlendem/zu engem Zeitfenster nichts geladen wurde und
        # keine Preloads gesetzt waren, laden wir sicherheitshalber alles.
        if not cim_snapshots and preloaded_snapshots is None:
            cim_snapshots = load_cim_snapshots(cim_root)

    # ---------------------------------------------------------
    # 6) Nur benötigte States preprocessen
    # ---------------------------------------------------------
    snapshot_cache, required_state_types = _build_request_snapshot_cache(
        cim_snapshots=cim_snapshots,
        parsed_query=parsed_query,
        analysis_plan=analysis_plan,
    )

    snapshot_cache_summary = summarize_snapshot_cache(snapshot_cache)

    # ---------------------------------------------------------
    # 7) Analyse
    #    Aktuell nutzt handle_user_query(...) intern weiterhin eigenes Parsing.
    #    Das ist okay für diesen Schritt; parsed_query / plan geben wir
    #    fürs Debug schon mit zurück.
    # ---------------------------------------------------------
    answer = handle_user_query(
    user_input=user_input,
    snapshot_cache=snapshot_cache,
    network_index=network_index,
    parsed_query=parsed_query,
    analysis_plan=analysis_plan,
)

    # ---------------------------------------------------------
    # 8) Standardisierte Rückgabe
    #    MCP-/Tool-freundlich: strukturierte Debug-Metadaten
    # ---------------------------------------------------------
    return {
        "status": "ok",
        "tool": "historical_cim_analysis",
        "input": user_input,
        "answer": answer,
        "debug": {
            "num_inventory_snapshots": len(snapshot_inventory.get("snapshots", [])) if snapshot_inventory else 0,
            "index_source_snapshot": network_index.get("index_source_snapshot"),
            "index_source_time_str": network_index.get("index_source_time_str"),
            "analysis_plan": analysis_plan,
            "planner_context": planner_context,
            "required_state_types": required_state_types,
            "num_loaded_snapshots": len(cim_snapshots),
            "loaded_snapshot_names": list(cim_snapshots.keys()),
            "snapshot_cache_summary": snapshot_cache_summary,
            "parsed_query": parsed_query,
        },
    }


# Optionaler Rückwärtskompatibilitäts-Wrapper für einfache Nutzung
def run(user_input: str, cim_root: str):
    """
    Einfache Kurzform für bestehende Aufrufer.
    """
    return run_historical_cim_analysis(
        user_input=user_input,
        cim_root=cim_root,
    )