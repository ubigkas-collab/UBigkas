import logging
import re
import os
import nltk
import torch
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize

# 1. INTEGRATION: Import custom components
# Ensure ubigkas_processor.py and marker_roberta.py are in the same folder
try:
    from ubigkas_processor import UBigkasProcessor
except ImportError:
    logging.warning("ubigkas_processor.py not found. Spelling correction will be disabled.")
    class UBigkasProcessor:
        def __init__(self, model_path): pass
        def process_sentence(self, text): return text

try:
    # This imports the logic from your marker_roberta.py file
    from marker_roberta import predict_tags, insert_markers
    HAS_MARKER_MODEL = True
except ImportError:
    logging.warning("marker_roberta.py not found in the current directory.")
    # Fallback dummies if file is missing
    def predict_tags(text): return text.split(), ["O"] * len(text.split()), [1.0] * len(text.split())
    def insert_markers(tokens, tags, scores): return " ".join(tokens)
    HAS_MARKER_MODEL = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FilipinoGrammarCorrector:
    """
    Hybrid Pipeline for Filipino Grammar Correction:
    1. ORIGINAL -> 2. CLEANED -> 3. TAGGED/FIXED -> 4. BRIDGE (EN) -> 5. FINAL (TL)
    """

    def __init__(self,
                 # UPDATE 1: Point to your new Fine-Tuned TL-EN model on Hugging Face
                 tl_en_model="Vinci14/final_tl_en_translator",
                 # UPDATE 2: Point to your existing Fine-Tuned EN-TL model on Hugging Face
                 en_tl_model="Vinci14/final_tagalog_translator", 
                 # UPDATE 3: Point to your Spelling model
                 spelling_model_path="Vinci14/my_spelling_model"): 
        
        # Setup NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        self.tl_en_model_name = tl_en_model
        self.en_tl_model_name = en_tl_model
        self.spelling_model_path = spelling_model_path
        
        # Load Translation components
        self._load_models()
        
        # Load Spelling components
        logger.info(f"Initializing UBigkas with: {self.spelling_model_path}")
        self.ubigkas = UBigkasProcessor(model_path=self.spelling_model_path)

    def _load_models(self):
        try:
            # --- LOAD TL-EN MODEL (Tagalog -> English Bridge) ---
            logger.info(f"Loading Fine-Tuned TL-EN Bridge from: {self.tl_en_model_name}")
            try:
                self.tl_en_tokenizer = MarianTokenizer.from_pretrained(self.tl_en_model_name)
                self.tl_en_model = MarianMTModel.from_pretrained(self.tl_en_model_name)
            except Exception as e:
                logger.warning(f"Failed to load custom TL-EN model. Fallback to generic: {e}")
                self.tl_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tl-en")
                self.tl_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tl-en")

            # --- LOAD EN-TL MODEL (English -> Tagalog Correction) ---
            logger.info(f"Loading Fine-Tuned EN-TL Correction Model from: {self.en_tl_model_name}")
            try:
                self.en_tl_tokenizer = MarianTokenizer.from_pretrained(self.en_tl_model_name)
                self.en_tl_model = MarianMTModel.from_pretrained(self.en_tl_model_name)
            except Exception as e:
                logger.warning(f"Failed to load custom EN-TL model. Fallback to generic: {e}")
                self.en_tl_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-tl")
                self.en_tl_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-tl")

            if HAS_MARKER_MODEL:
                logger.info("Marker RoBERTa logic detected and ready for Stage 3.")

        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            raise

    def translate_tl_to_en(self, text):
        inputs = self.tl_en_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = self.tl_en_model.generate(**inputs)
        return self.tl_en_tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_en_to_tl(self, text):
        inputs = self.en_tl_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        # Using beams=4 for higher quality during reconstruction
        translated = self.en_tl_model.generate(**inputs, max_length=512, num_beams=4)
        return self.en_tl_tokenizer.decode(translated[0], skip_special_tokens=True)

    def _refine_english(self, text):
        # Normalize the English bridge to help the decoder
        text = re.sub(r"\bI'm\b", "I am", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcan't\b", "cannot", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdon't\b", "do not", text, flags=re.IGNORECASE)
        return text

    def _post_process_filipino(self, text):
        text = text.strip().capitalize()
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        return text

    def _process_single_sentence(self, sentence):
        """Helper to process one sentence at a time."""
        # 1. CLEANED (Spelling Fixes)
        cleaned = self.ubigkas.process_sentence(sentence)

        # 2. TAGGED/FIXED (RoBERTa Tagging + Marker Insertion + Conjugation)
        if HAS_MARKER_MODEL:
            try:
                # Capture tokens, tags, AND SCORES (The Fix)
                tokens, tags, scores = predict_tags(cleaned)
                # Pass ALL THREE to insert_markers
                tagged = insert_markers(tokens, tags, scores)
            except ValueError as e:
                logger.error(f"Mismatch in RoBERTa output: {e}")
                tagged = cleaned
        else:
            tagged = cleaned
        
        # 3. BRIDGE (EN)
        english_raw = self.translate_tl_to_en(tagged)
        bridge = self._refine_english(english_raw)

        # 4. FINAL (TL)
        marian_raw = self.translate_en_to_tl(bridge)
        final = self._post_process_filipino(marian_raw)
        
        return cleaned, tagged, bridge, final

    def correct_grammar_with_pipeline(self, text):
        """ Runs the full hybrid pipeline on multiple sentences. """
        if not text.strip(): return
        
        # Split input into sentences
        sentences = sent_tokenize(text.strip())
        
        final_output_parts = []
        
        print("\n" + "="*80)
        print(f"{'PIPELINE STAGE':<20} | {'CONTENT'}")
        print("-" * 80)
        print(f"{'INPUT TEXT':<20} | {text}")
        print("-" * 80)

        for i, sentence in enumerate(sentences):
            cleaned, tagged, bridge, final = self._process_single_sentence(sentence)
            
            # Print details for this sentence
            prefix = f"[Sent {i+1}] "
            print(f"{prefix + 'CLEANED':<20} | {cleaned}")
            print(f"{prefix + 'TAGGED':<20} | {tagged}")
            print(f"{prefix + 'BRIDGE':<20} | {bridge}")
            print(f"{prefix + 'FINAL':<20} | {final}")
            print("-" * 80)
            
            final_output_parts.append(final)

        full_final_output = " ".join(final_output_parts)
        print(f"{'FULL OUTPUT':<20} | {full_final_output}")
        print("="*80 + "\n")

        return full_final_output

    def interactive_mode(self):
        print("--- Filipino Grammar Corrector (V3 - Dual Fine-Tuned Models) ---")
        print("Type 'exit' to quit.")
        while True:
            try:
                text = input("Enter text: ").strip()
                if text.lower() in ["exit", "quit"]: break
                self.correct_grammar_with_pipeline(text)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing input: {e}")

if __name__ == "__main__":
    corrector = FilipinoGrammarCorrector()
    corrector.interactive_mode()