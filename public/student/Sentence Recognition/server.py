from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os
import logging
import nltk

# Auto-download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
nlp_path = os.path.abspath(os.path.join(current_dir, '..', 'NLP'))

if nlp_path not in sys.path:
    sys.path.append(nlp_path)

# --- NLP Imports ---
try:
    from filipino_grammar_corrector import FilipinoGrammarCorrector
    from filipino_rules import analyze_word, detect_sentence_structure
    from dictionary_utils import get_meaning_and_type
except ImportError as e:
    logger.critical(f"Failed to import NLP modules. Error: {e}")
    sys.exit(1)

app = Flask(__name__)
# Enhanced CORS to handle pre-flight OPTIONS requests
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Initialize AI Models ---
logger.info("Initializing New Transformer-Based Filipino Grammar Corrector...")
tl_en_model_path = os.path.join(nlp_path, 'final_tl_en_translator')
en_tl_model_path = os.path.join(nlp_path, 'final_tagalog_translator')
spelling_model_path = os.path.join(nlp_path, 'my_spelling_model')

try:
    corrector = FilipinoGrammarCorrector(
        tl_en_model=tl_en_model_path,
        en_tl_model=en_tl_model_path,
        spelling_model_path=spelling_model_path
    )
    logger.info("AI Models loaded successfully!")
except Exception as e:
    logger.critical(f"Failed to initialize AI models: {e}")
    sys.exit(1)

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_sentence():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        structure = detect_sentence_structure(sentence)
        raw_words = re.findall(r"\b[\w-]+\b", sentence)
        word_details = []
        
        for word in raw_words:
            meaning, word_type, _, _, _, _ = get_meaning_and_type(word)
            info = analyze_word(word)
            info['word'] = word
            info['meaning'] = meaning if meaning else "No meaning found"
            info['type'] = word_type if word_type else "Unknown"
            word_details.append(info)

        return jsonify({
            "original": sentence,
            "corrected": sentence,
            "structure": structure,
            "words": word_details
        })
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/correct", methods=["POST", "OPTIONS"])
def correct_sentence():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        corrected_text = corrector.correct_grammar_with_pipeline(sentence)
        return jsonify({
            "original": sentence,
            "corrected": corrected_text
        })
    except Exception as e:
        logger.error(f"Correction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)