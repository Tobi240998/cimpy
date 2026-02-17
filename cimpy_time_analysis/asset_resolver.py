import re
from difflib import get_close_matches


def normalize_text(text: str) -> str:
    """
    Aggressive Normalisierung für robustes Matching:
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
    t = re.sub(r"\s+", "", t)
    return t


def extract_two_numbers(text: str):
    """
    Extrahiert zwei Zahlen aus Text, z.B.:
      "Trafo 19 - 20" -> ("19", "20")
      "Transformator 19/20" -> ("19", "20")
    """
    if not text:
        return None

    m = re.search(r"(\d+)\D+(\d+)", text)
    if not m:
        return None

    return m.group(1), m.group(2)


def resolve_equipment_from_query(
    user_input: str,
    equipment_type: str,
    network_index: dict,
    cutoff: float = 0.65
):
    """
    Liefert (equipment_obj, debug_info) oder (None, debug_info)

    Matching-Reihenfolge:
    1) Direkter Match gegen normalisierte Namen
    2) Nummern-basiertes Match (z.B. 19 und 20 müssen im Namen vorkommen)
    3) Fuzzy Match gegen normalisierte Namen (difflib)
    """

    debug = {
        "equipment_type": equipment_type,
        "user_input": user_input,
        "normalized_user_input": normalize_text(user_input),
        "method": None,
        "matched_name": None
    }

    name_index = network_index.get("equipment_name_index", {}).get(equipment_type, {})
    if not name_index:
        debug["method"] = "no_index"
        return None, debug

    user_norm = debug["normalized_user_input"]

    # 1) Direkter Treffer
    if user_norm in name_index:
        eq = name_index[user_norm]
        debug["method"] = "direct_normalized"
        debug["matched_name"] = getattr(eq, "name", None)
        return eq, debug

    # 2) Nummern-Extraktion (19/20 etc.)
    nums = extract_two_numbers(user_input)
    if nums:
        n1, n2 = nums
        for norm_name, eq in name_index.items():
            # sehr robust: beide Zahlen müssen irgendwo im normalisierten Namen vorkommen
            if n1 in norm_name and n2 in norm_name:
                debug["method"] = "number_match"
                debug["matched_name"] = getattr(eq, "name", None)
                return eq, debug

    # 3) Fuzzy
    matches = get_close_matches(user_norm, list(name_index.keys()), n=1, cutoff=cutoff)
    if matches:
        eq = name_index[matches[0]]
        debug["method"] = "fuzzy"
        debug["matched_name"] = getattr(eq, "name", None)
        return eq, debug

    debug["method"] = "no_match"
    return None, debug
