from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os
import logging

# Configure logging so Flask logs and the corrector logs show up together
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Configuration ---
# Get the directory where server.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the NLP folder (one level up)
nlp_path = os.path.abspath(os.path.join(current_dir, '..', 'NLP'))

# Add nlp_path to sys.path so we can import modules from it
if nlp_path not in sys.path:
    sys.path.append(nlp_path)

logger.info(f"NLP Path set to: {nlp_path}")

# --- NLP Imports ---
try:
    # Import the new class from the updated file in ../NLP/
    from filipino_grammar_corrector import FilipinoGrammarCorrector
    # Assuming these still exist in ../NLP/ for the /analyze route
    from filipino_rules import analyze_word, detect_sentence_structure
    from dictionary_utils import get_meaning_and_type
except ImportError as e:
    logger.critical(f"Failed to import NLP modules from {nlp_path}. Error: {e}")
    sys.exit(1)


app = Flask(__name__)
CORS(app)  # Allow frontend to access this server

# --- Initialize AI Models ---
logger.info("Initializing New Transformer-Based Filipino Grammar Corrector...")

# Define absolute paths to the model folders inside the NLP directory
tl_en_model_path = os.path.join(nlp_path, 'final_tl_en_translator')
en_tl_model_path = os.path.join(nlp_path, 'final_tagalog_translator')
spelling_model_path = os.path.join(nlp_path, 'my_spelling_model')

try:
    # Initialize the corrector with model paths instead of text files
    corrector = FilipinoGrammarCorrector(
        tl_en_model=tl_en_model_path,
        en_tl_model=en_tl_model_path,
        spelling_model_path=spelling_model_path
    )
    logger.info("AI Models loaded successfully! Server is ready.")
except Exception as e:
    logger.critical(f"Failed to initialize AI models. Server cannot start. Error: {e}")
    sys.exit(1)


@app.route("/analyze", methods=["POST"])
def analyze_sentence():
    """
    Analyzes sentence structure and word types using rule-based logic.
    (This route remains unchanged based on your provided server code).
    """
    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # --- Pre-check for simple dictionary corrections ---
    words = re.findall(r"\b[\w-]+\b", sentence)
    suggested_corrections = []
    
    for word in words:
        # Check dictionary utils for basic corrections
        meaning, word_type, suggested_word, corrected, affix, affix_explanation = get_meaning_and_type(word)
        if corrected and suggested_word != word:
            suggested_corrections.append((word, suggested_word))

    # Apply dictionary-based fixes first
    corrected_words = [sugg if corrected and sugg != word else word for word, sugg in suggested_corrections]
    corrected_sentence_base = " ".join(corrected_words) if corrected_words else sentence

    # --- Analyze sentence structure ---
    try:
        structure = detect_sentence_structure(corrected_sentence_base)
    except Exception as e:
        logger.error(f"Error during structure detection: {e}")
        structure = "Analysis Failed"
    
    # --- Analyze individual words ---
    word_details = []
    for word in corrected_sentence_base.split():
        info = analyze_word(word)
        word_details.append(info)

    return jsonify({
        "original": sentence,
        "corrected_base": corrected_sentence_base,
        "structure": structure,
        "words": word_details
    })

@app.route("/correct", methods=["POST"])
def correct_sentence():
    """
    Uses the new AI pipeline (UBigkas + RoBERTa + MarianMT) to correct grammar.
    """
    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    logger.info(f"Processing correction request length: {len(sentence)}")

    try:
        # Run the new full AI pipeline.
        # NOTE: The intermediate steps (cleaned, tagged, bridge) will be printed 
        # to the server console logs by the corrector class itself.
        corrected_text = corrector.correct_grammar_with_pipeline(sentence)
        
        # The new pipeline method only returns the final string.
        # We update the JSON response to match available data.
        return jsonify({
            "original": sentence,
            "corrected": corrected_text
        })
        
    except Exception as e:
        logger.error(f"Error processing sentence in AI pipeline: {e}", exc_info=True)
        return jsonify({"error": f"Internal Error: {str(e)}"}), 500

if __name__ == "__main__":
    # Run on port 5000
    # use_reloader=False is recommended when loading heavy models to prevent double-loading
    app.run(debug=True, port=5000, use_reloader=False)