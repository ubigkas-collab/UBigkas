import json
import re
import difflib
import os

# Load dictionary
with open(os.path.join(os.path.dirname(__file__), "tagalog_dictionary.json"), "r", encoding="utf-8") as f:
    TAGALOG_DICT = json.load(f)

VALID_WORDS = set(entry["word"].lower() for entry in TAGALOG_DICT)

# POS pattern
TYPE_PATTERN = re.compile(r"(n\.|v\.|adj\.|gram\.|intrj\.|prep\.|adv\.)\s*(.*)", re.IGNORECASE)
TYPE_MAP = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "gram": "grammar/particle",
    "intrj": "interjection",
    "prep": "preposition",
    "adv": "adverb"
}

# Common verb affixes for inference
VERB_PREFIXES = ["mag", "nag", "um", "ma", "ka"]
VERB_SUFFIXES = ["in", "an", "i"]

# Known markers/particles not to treat as verbs
MARKERS = ["si", "sina", "ang", "mga", "ay", "ng", "ni", "nina", "sa", "kayo", "ako"]

def get_meaning_and_type(word):
    word_lower = word.lower()
    meaning = "No definition found."
    word_type = "unknown"
    suggested_word = word
    corrected = False
    affix = None
    affix_explanation = None

    # --- Exact dictionary match ---
    matches = [entry for entry in TAGALOG_DICT if entry["word"].lower() == word_lower]
    if matches:
        entry = matches[0]
        definition = entry["definition"].strip()
        # Search POS anywhere in definition
        m = TYPE_PATTERN.search(definition)
        if m:
            code = m.group(1).replace(".", "").lower()
            word_type = TYPE_MAP.get(code, "unknown")
            meaning = m.group(2).strip()
        else:
            word_type = "unknown"
            meaning = definition
    else:
        # --- Auto-correct suggestion ---
        candidates = [entry["word"] for entry in TAGALOG_DICT]
        closest = difflib.get_close_matches(word, candidates, n=1, cutoff=0.8)
        if closest:
            suggested_word = closest[0]
            corrected = True
            matches = [entry for entry in TAGALOG_DICT if entry["word"] == suggested_word]
            if matches:
                entry = matches[0]
                definition = entry["definition"].strip()
                m = TYPE_PATTERN.search(definition)
                if m:
                    code = m.group(1).replace(".", "").lower()
                    word_type = TYPE_MAP.get(code, "unknown")
                    meaning = m.group(2).strip()
                else:
                    word_type = "unknown"
                    meaning = definition

    # --- Infer verb type from affix ONLY if unknown and not a marker ---
    if word_type == "unknown" and word_lower not in MARKERS:
        inferred = False
        for prefix in VERB_PREFIXES:
            if word_lower.startswith(prefix):
                word_type = "verb"
                inferred = True
                affix = prefix
                break
        if not inferred:
            for suffix in VERB_SUFFIXES:
                if word_lower.endswith(suffix):
                    word_type = "verb"
                    inferred = True
                    affix = suffix
                    break
        if inferred:
            # Simple affix explanation (can be extended)
            if affix in ["mag", "nag"]:
                affix_explanation = "Actor focus; indicates progressive/future tense."
            elif affix == "um":
                affix_explanation = "Actor focus; often indicates completed or recent action."
            elif affix == "ma":
                affix_explanation = "Stative verbs or adjectives; describes a state or condition."
            elif affix == "ka":
                affix_explanation = "Reciprocal or collective focus."
            elif affix in ["in", "an", "i"]:
                affix_explanation = "Object or locative focus; indicates completed or imperative aspect."

    return meaning, word_type, suggested_word, corrected, affix, affix_explanation
