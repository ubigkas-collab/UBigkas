from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os

# -------------------------------------------------
# PROJECT ROOT (for NLP imports)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from NLP.filipino_grammar_corrector import FilipinoGrammarCorrector
from NLP.filipino_rules import analyze_word, detect_sentence_structure
from NLP.dictionary_utils import get_meaning_and_type

# -------------------------------------------------
# FLASK APP
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# MODEL INITIALIZATION
# -------------------------------------------------
# Lazy loading: initializes once when server starts
NLP_DIR = os.path.join(PROJECT_ROOT, "NLP")

corrector = FilipinoGrammarCorrector(
    dictionary_path=os.path.join(NLP_DIR, "Filipino-wordlist.txt"),
    slang_path=os.path.join(NLP_DIR, "slang_map.txt")
)

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze_sentence():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        result = corrector.correct_grammar(sentence)
        corrected = result["marian_corrected"]

        words = re.findall(r"\b[\w-]+\b", corrected)
        word_details = [analyze_word(w) for w in words]
        structure = detect_sentence_structure(corrected)

        return jsonify({
            "original": sentence,
            "corrected": corrected,
            "structure": structure,
            "words": word_details,
            "english_translation": result["english"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correct", methods=["POST"])
def correct_sentence():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        result = corrector.correct_grammar(sentence)
        return jsonify({
            "original": result["original"],
            "corrected": result["marian_corrected"],
            "slang_removed": result["cleaned"],
            "english_translation": result["english"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# ENTRYPOINT FOR RENDER
# -------------------------------------------------
if __name__ == "__main__":
    # RENDER sets PORT in env
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
