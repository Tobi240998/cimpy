import re
from difflib import get_close_matches


def normalize_text(text: str) -> str:
    """
    Normalisierung für Matching:
    - lower
    - transformator -> trafo
    - trf -> trafo
    - / -> -
    - Leerzeichen entfernen
    """
    if not text:
        return ""

    t = text.lower()
    t = t.replace("transformator", "trafo")
    t = t.replace("trf", "trafo")
    t = t.replace("/", "-")
    t = re.sub(r"\s+", "", t) #Entfernen von Whitespace-Zeichen (Leerzeichen, Tab, Zeilenumbruch, ...)
    return t


def extract_two_numbers(text: str):
    """
    Extrahiert zwei Zahlen aus Text, z.B.:
      "Trafo 19 - 20" -> ("19", "20")
      "Transformator 19/20" -> ("19", "20")
    """
    if not text:
        return None

    m = re.search(r"(\d+)\D+(\d+)", text) #sucht nach zusammenhängender Zahlenfolge und speichert sie, überspringt dann "Nicht-Zahlen" und speichert nächste vorkommende Zahlenfolge wieder
    if not m:
        return None

    return m.group(1), m.group(2)


def extract_one_number(text: str):
    """
    Extrahiert eine Zahl aus Text, z.B.:
      "Load 27" -> "27"
      "Verbraucher 08" -> "08"
    Achtung: Wenn mehrere Zahlen vorkommen, nehmen wir die erste.
    """
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    return m.group(1)


def _number_boundary_match(num: str, norm_name: str) -> bool:
    """
    Prüft, ob 'num' als eigenständige Zahl im normalisierten Namen vorkommt,
    so dass z.B. 27 nicht auf 127 matcht.
    """
    if not num or not norm_name:
        return False

    # Boundary-Regel: vor/nach der Zahl darf keine weitere Ziffer stehen
    # Beispiel: "load27" -> match, "load127" -> kein match für 27
    pattern = rf"(^|[^0-9]){re.escape(num)}([^0-9]|$)"
    return re.search(pattern, norm_name) is not None


def resolve_equipment_from_query(
    user_input: str,
    equipment_type: str | None,   #kann None sein, falls Typ nicht bekannt ist -> dann wird über alle Typen gesucht
    network_index: dict,
    cutoff: float = 0.65
):
    """
    Liefert (equipment_obj, debug_info) oder (None, debug_info)

    Matching-Reihenfolge:
    1) Direkter Match gegen normalisierte Namen (Substring-Check, da User oft ganze Sätze schreibt)
    2) Nummern-basiertes Match mit zwei Zahlen (z.B. 19 und 20 müssen im Namen vorkommen) -> gut für Trafos
    3) Nummern-basiertes Match mit einer Zahl (z.B. Load 27) -> gut für Lasten/Verbraucher
    4) Fuzzy Match gegen normalisierte Namen (difflib)
    """
    #zur Nachvollziehbarkeit, was wie gematcht wurde
    debug = {
        "equipment_type": equipment_type,
        "user_input": user_input,
        "normalized_user_input": normalize_text(user_input),
        "method": None,
        "matched_name": None,
        "matched_type": None
    }

    print(list(network_index["equipment_name_index"]["ConformLoad"].keys())[:20])

    equipment_name_index = network_index.get("equipment_name_index", {})
    if not equipment_name_index:
        debug["method"] = "no_index"
        return None, debug

    user_norm = debug["normalized_user_input"]

    #Verfügbare Namen werden aus Network Index geholt
    #Wenn equipment_type bekannt ist, wird nur in diesem Typ gesucht, sonst in allen vorhandenen Typen (z.B. PowerTransformer, EnergyConsumer, ...)
    if equipment_type:
        type_spaces = [(equipment_type, equipment_name_index.get(equipment_type, {}))]
    else:
        type_spaces = [(t, idx) for t, idx in equipment_name_index.items() if idx]

    if not type_spaces or all(not idx for _, idx in type_spaces):
        debug["method"] = "no_index"
        return None, debug

    # 1) Direkter Treffer (Substring) - damit ganze Sätze matchen können
    # Prüfung, ob ein normalisierter Equipment-Name als Teilstring in der normalisierten User-Eingabe vorkommt
    best_direct = None  # (type, norm_name, eq, score)
    for t, name_index in type_spaces:
        for norm_name, eq in name_index.items():
            if norm_name and norm_name in user_norm:
                # Längerer Treffer ist meist spezifischer ("load27" > "load2")
                score = len(norm_name)
                if best_direct is None or score > best_direct[3]:
                    best_direct = (t, norm_name, eq, score)

    if best_direct:
        t, norm_name, eq, _ = best_direct
        debug["method"] = "direct_substring"
        debug["matched_name"] = getattr(eq, "name", None)
        debug["matched_type"] = t
        return eq, debug

    # 2) Nummern-Extraktion (19/20 etc.) -> sehr robust für Trafos
    nums = extract_two_numbers(user_input) #zieht, falls vorhanden, Zahlen aus dem User-Input
    if nums:
        n1, n2 = nums
        for t, name_index in type_spaces:
            for norm_name, eq in name_index.items():
                #beide Zahlen müssen irgendwo im normalisierten Namen vorkommen
                if n1 in norm_name and n2 in norm_name:
                    debug["method"] = "number_match_two"
                    debug["matched_name"] = getattr(eq, "name", None)
                    debug["matched_type"] = t
                    return eq, debug

    # 3) Single-Number-Match (Load 27 etc.) -> vor Fuzzy, weil deterministischer
    n = extract_one_number(user_input)
    if n:
        for norm_name, eq in name_index.items():
            #Zahl muss als eigenständige Zahl vorkommen (27 matcht nicht auf 127)
            if _number_boundary_match(n, norm_name):
                debug["method"] = "number_match_one"
                debug["matched_name"] = getattr(eq, "name", None)
                return eq, debug

    # 4) Fuzzy - sucht nach Textähnlichkeit, um Tippfehler etc. auszuschließen (deterministischer Ähnlichkeitsvergleich)
    best_fuzzy = None  # (type, match_key, eq)
    for t, name_index in type_spaces:
        matches = get_close_matches(user_norm, list(name_index.keys()), n=1, cutoff=cutoff)
        if matches:
            match_key = matches[0]
            eq = name_index[match_key]
            # falls mehrere Typen matchen, nehmen wir den spezifischeren (längerer Key)
            if best_fuzzy is None or len(match_key) > len(best_fuzzy[1]):
                best_fuzzy = (t, match_key, eq)

    if best_fuzzy:
        t, match_key, eq = best_fuzzy
        debug["method"] = "fuzzy"
        debug["matched_name"] = getattr(eq, "name", None)
        debug["matched_type"] = t
        return eq, debug

    debug["method"] = "no_match"
    return None, debug
