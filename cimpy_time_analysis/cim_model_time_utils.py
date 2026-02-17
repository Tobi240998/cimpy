from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


MD_NS = "http://iec.ch/TC57/61970-552/ModelDescription/1#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


@dataclass(frozen=True)
class FullModelTimes:
    """
    Zeitinformationen aus md:FullModel.
    """
    profile: str | None
    created: datetime | None
    scenario_time: datetime | None
    source_file: str | None


def _parse_iso8601(dt_str: str) -> datetime | None:
    """
    Parst ISO8601 aus CGMES (oft mit 'Z').
    Gibt timezone-aware datetime zurück, wenn Offset vorhanden.
    """
    if not dt_str:
        return None

    s = dt_str.strip()
    # "Z" -> UTC Offset
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def extract_fullmodel_times_from_xml(xml_path: str | Path) -> FullModelTimes:
    """
    Liest md:Model.profile, md:Model.created, md:Model.scenarioTime aus einer CGMES XML.

    Wichtig:
    - Diese Infos stehen im md:FullModel Header und werden von CIMpy oft nicht als Objekt bereitgestellt.
    """
    xml_path = Path(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return FullModelTimes(profile=None, created=None, scenario_time=None, source_file=str(xml_path))

    # Suche md:FullModel (kann mit rdf:about oder rdf:ID kommen)
    full_models = root.findall(f".//{{{MD_NS}}}FullModel")
    if not full_models:
        return FullModelTimes(profile=None, created=None, scenario_time=None, source_file=str(xml_path))

    fm = full_models[0]

    def _find_text(tag_local: str) -> str | None:
        el = fm.find(f".//{{{MD_NS}}}{tag_local}")
        return el.text.strip() if (el is not None and el.text) else None

    profile = _find_text("Model.profile")
    created_str = _find_text("Model.created")
    scenario_str = _find_text("Model.scenarioTime")

    return FullModelTimes(
        profile=profile,
        created=_parse_iso8601(created_str) if created_str else None,
        scenario_time=_parse_iso8601(scenario_str) if scenario_str else None,
        source_file=str(xml_path),
    )


def choose_snapshot_scenario_time(xml_files: list[str]) -> FullModelTimes | None:
    """
    Wählt pro Snapshot den besten scenarioTime.

    Heuristik:
    1) Bevorzugt Datei mit md:Model.profile, die auf StateVariables (SV) hindeutet und scenarioTime hat.
    2) Sonst irgendeine Datei mit scenarioTime.
    3) Sonst None.

    Damit bist du unabhängig vom Dateinamen und nutzt ausschließlich CIM-Headerdaten.
    """
    candidates: list[FullModelTimes] = []
    for f in xml_files:
        candidates.append(extract_fullmodel_times_from_xml(f))

    # 1) SV bevorzugen (Profile-URL enthält typischerweise "StateVariables")
    sv_with_time = [
        c for c in candidates
        if c.scenario_time is not None and c.profile and "StateVariables" in c.profile
    ]
    if sv_with_time:
        # Falls mehrere: nimm die erste (oder könnte man nach created sortieren)
        return sv_with_time[0]

    # 2) irgendeine scenarioTime
    any_with_time = [c for c in candidates if c.scenario_time is not None]
    if any_with_time:
        return any_with_time[0]

    return None
