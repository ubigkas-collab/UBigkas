import re
from dictionary_utils import get_meaning_and_type, MARKERS

# --- Analyze a single word ---
def analyze_word(word):
    meaning, word_type, suggested_word, corrected, affix, affix_explanation = get_meaning_and_type(word)
    return {
        "word": word,
        "type": word_type,
        "meaning": meaning,
        "suggested_word": suggested_word,
        "corrected": corrected,
        "affix": affix,
        "affix_explanation": affix_explanation
    }

# --- Detect sentence structure ---
def detect_sentence_structure(sentence):
    words = re.findall(r"\b[\w-]+\b", sentence)
    if not words:
        return "Unknown pattern"

    analyzed = [analyze_word(w) for w in words]
    types = [w["type"] for w in analyzed]

    # Basic pattern rules
    pattern = []
    first_noun_found = False
    for t in types:
        if t == "verb":
            pattern.append("V")
        elif t == "noun":
            if not first_noun_found:
                pattern.append("S")
                first_noun_found = True
            else:
                pattern.append("O")
        elif t in ["grammar/particle", "marker"]:
            pattern.append("Link-P")
        elif t == "adjective":
            pattern.append("Adj")
        else:
            pattern.append("X")  # unknown/other

    pattern_str = "-".join(pattern)

    # Simplify common Filipino sentence structures
    if pattern_str.startswith("V-S"):
        return "VSO (Verb-Subject-Object)"
    elif pattern_str.startswith("S-Link-P"):
        return "S-Link-P (ay-inverted)"
    elif pattern_str.startswith("S-V"):
        return "SVO (Subject-Verb-Object)"
    else:
        return "Unknown pattern"
