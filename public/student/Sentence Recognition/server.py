from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
nlp_path = os.path.join(current_dir, '..', 'NLP')
sys.path.append(nlp_path)

# --- NLP Imports ---
from filipino_grammar_corrector import FilipinoGrammarCorrector
from filipino_rules import analyze_word, detect_sentence_structure
from dictionary_utils import get_meaning_and_type

app = Flask(__name__)
CORS(app)

# --- Initialize AI Models ---
print("Initializing Filipino Grammar Corrector...")
corrector = FilipinoGrammarCorrector(
    dictionary_path=os.path.join(nlp_path, 'Filipino-wordlist.txt'),
    slang_path=os.path.join(nlp_path, 'slang_map.txt')
)
print("Server is ready!")

@app.route("/analyze", methods=["POST"])
def analyze_sentence():
    data = request.get_json()
    # The frontend should now be sending the AI-corrected sentence here
    sentence = data.get("sentence", "").strip()
    
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # 1. Split the already-corrected sentence into words
    # We use the sentence as provided by the user/frontend
    raw_words = re.findall(r"\b[\w-]+\b", sentence)
    word_details = []
    
    for word in raw_words:
        # 2. Just get the data for this specific word
        meaning, word_type, _, _, _, _ = get_meaning_and_type(word)
        
        # 3. Perform grammar rule analysis (roots, affixes, etc.)
        info = analyze_word(word)
        
        # 4. Populate the table data
        info['word'] = word  # Ensure the word itself is included
        info['meaning'] = meaning if meaning else "Walang kahulugan"
        info['type'] = word_type if word_type else "Di-tukoy"
        word_details.append(info)

    # 5. Detect structure of this specific sentence
    structure = detect_sentence_structure(sentence)

    return jsonify({
        "original": sentence, # This is the "corrected" sentence sent by frontend
        "corrected": sentence, 
        "structure": structure,
        "words": word_details
    })

@app.route("/correct", methods=["POST"])
def correct_sentence():
    """
    Uses the AI model to return the final corrected version.
    """
    data = request.get_json()
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
        print(f"Error processing sentence: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)