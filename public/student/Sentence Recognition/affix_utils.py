import json
from dictionary_utils import VALID_WORDS

# Load affix rules
with open("affix_rules.json", "r", encoding="utf-8") as f:
    VERB_AFFIXES = json.load(f)

def detect_affix(word, word_type):
    word_lower = word.lower()

    if word_type not in ["verb", "unknown"]:
        return None, None, None  # no affix

    # Check prefixes
    for affix in ["mag", "nag", "na", "i", "ma", "ka"]:
        if word_lower.startswith(affix):
            root = word_lower[len(affix):]
            if root in VALID_WORDS:
                data = VERB_AFFIXES.get(affix, {})
                return affix, data.get("explanation", ""), data.get("note", "")

    # Check suffixes
    for affix in ["in", "an"]:
        if word_lower.endswith(affix):
            root = word_lower[:-len(affix)]
            if root in VALID_WORDS:
                data = VERB_AFFIXES.get(affix, {})
                return affix, data.get("explanation", ""), data.get("note", "")

    # Infix "um"
    if len(word_lower) >= 3 and "um" in word_lower[1:3]:
        root = word_lower.replace("um", "", 1)
        if root in VALID_WORDS:
            data = VERB_AFFIXES.get("um", {})
            return "um", data.get("explanation", ""), data.get("note", "")

    return None, None, None
