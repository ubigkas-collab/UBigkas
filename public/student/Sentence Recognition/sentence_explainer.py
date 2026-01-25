# sentence_explainer.py
import re
from filipino_rules import analyze_word, detect_sentence_structure
from dictionary_utils import get_meaning_and_type

def analyze_sentence(sentence):
    words = re.findall(r"\b[\w-]+\b", sentence)
    suggested_corrections = []

    for word in words:
        meaning, word_type, suggested_word, corrected, affix, affix_explanation = get_meaning_and_type(word)
        if corrected and suggested_word != word:
            suggested_corrections.append((word, suggested_word))

    # Auto-apply corrections for simplicity
    corrected_words = [suggested_word if any(w == orig for orig, sugg in suggested_corrections) else w for w in words]
    corrected_sentence = " ".join(corrected_words)

    structure = detect_sentence_structure(corrected_sentence)
    word_analysis = []

    for word in corrected_sentence.split():
        info = analyze_word(word)
        word_analysis.append({
            "word": info['word'],
            "type": info['type'],
            "meaning": info['meaning'],
            "affix": info.get('affix'),
            "affix_explanation": info.get('affix_explanation'),
            "corrected": info.get('corrected', False),
            "suggested_word": info.get('suggested_word')
        })

    return {
        "original_sentence": sentence,
        "corrected_sentence": corrected_sentence if corrected_sentence != sentence else None,
        "structure": structure,
        "words": word_analysis
    }
