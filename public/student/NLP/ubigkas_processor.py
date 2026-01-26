import re
import os
import torch
import logging
from collections import Counter
from spellchecker import SpellChecker
from transformers import RobertaTokenizer, RobertaForMaskedLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UBigkasProcessor:
    # UPDATED: Points to the Hugging Face Hub repository
    def __init__(self, model_path="Vinci14/my_spelling_model"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Ensure these files exist in the same directory as the script
        dictionary_path = os.path.join(base_dir, 'Filipino-wordlist.txt')
        slang_path = os.path.join(base_dir, 'slang_map.txt')

        self.spell = SpellChecker(language=None, distance=2) 
        
        # 1. Load Wordlist
        self.word_list = []
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                self.word_list = [line.strip().lower() for line in f if line.strip()]
            self.spell.word_frequency.load_words(self.word_list)
            logger.info(f"✅ Dictionary Loaded: {len(self.word_list)} words")
        else:
            logger.warning(f"⚠️ Warning: Wordlist not found at {dictionary_path}")

        # 2. Load Slang Map
        self.slang_map = {}
        if os.path.exists(slang_path):
            with open(slang_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        self.slang_map[key.strip().lower()] = value.strip()
        
        # 3. Finalized Overrides
        self.typo_overrides = {
            'gnto': ['ganito', 'ginto'],
            'pntahan': ['puntahan', 'pintahan'],
            'pnta': ['punta', 'pinta'],
            'bhy': ['bahay', 'buhay'],
            'bhay': ['bahay', 'buhay'],
        }

        # 4. Finalized Triggers
        self.jewelry_words = {'singsing', 'kwintas', 'kuwintas', 'hikaw', 'ginto', 'alahas', 'suot', 'presyo'}
        self.paint_words = {'pader', 'dingding', 'kulay', 'pintura', 'pula', 'asul', 'berde', 'dilaw', 'puti', 'itim'}

        # 5. Load Fine-Tuned Brain (Updated for Hugging Face)
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🔄 Attempting to load model from: {model_path}...")
        try:
            # Removed 'local_files_only=True' to allow downloading from Hub
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaForMaskedLM.from_pretrained(model_path)
            self.model.eval()
            logger.info("✅ Context Brain (Fine-Tuned) Loaded Successfully")
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            logger.info("⚠️ System will continue using only dictionary/edit-distance logic.")

    def _tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)

    def _levenshtein(self, s1, s2):
        if len(s1) < len(s2): return self._levenshtein(s2, s1)
        if len(s2) == 0: return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
            prev = curr
        return prev[-1]

    def get_candidates(self, word, context_words=None):
        word_lower = word.lower()
        if word_lower in self.spell.known([word_lower]): return [word] 

        # Context Trigger Logic
        if context_words:
            if word_lower == 'gnto' and any(j in context_words for j in self.jewelry_words):
                return ['ginto']
            if word_lower in ['pntahan', 'pnta'] and any(p in context_words for p in self.paint_words):
                return ['pintahan'] if word_lower == 'pntahan' else ['pinta']

        # Override Logic
        if word_lower in self.typo_overrides:
            valid = [c for c in self.typo_overrides[word_lower] if c in self.word_list]
            if valid: return valid

        # Standard Math Fallback
        candidates = [w for w in self.word_list if w.startswith(word_lower[0])]
        possible = [w for w in candidates if abs(len(word_lower) - len(w)) <= 3 and self._levenshtein(word_lower, w) <= 2]

        if not possible:
            corr = self.spell.correction(word_lower)
            return [corr] if corr else [word]

        possible.sort(key=lambda x: (self._levenshtein(word_lower, x), -self.spell.word_frequency.dictionary.get(x, 1)))
        return possible[:15]

    def _rank_with_bert(self, tokens, idx, candidates):
        if not self.model: return candidates[0]
        try:
            prefix, suffix = "".join(tokens[:idx]), "".join(tokens[idx+1:])
            # Insert mask token properly
            masked = f"{prefix}{self.tokenizer.mask_token}{suffix}"
            inputs = self.tokenizer(masked, return_tensors="pt")
            
            # Find the index of the mask token
            m_idx = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            best_c, best_s = candidates[0], -float('inf')
            
            for c in candidates:
                # Tokenize candidate to check if it's a single token or subwords
                # Ideally, we check the probability of the first token if it splits, 
                # but for simplicity in spelling correction, we assume single-token replacement overlap.
                c_id = self.tokenizer.convert_tokens_to_ids(c)
                
                # Skip unknown tokens
                if c_id == self.tokenizer.unk_token_id: continue
                
                score = logits[0, m_idx, c_id].item()
                if score > best_s: best_s, best_c = score, c
            return best_c
        except Exception as e:
            # logger.debug(f"BERT ranking error: {e}")
            return candidates[0]

    def process_sentence(self, text):
        text = self.normalize_slang(text)
        tokens = self._tokenize(text)
        
        # Build context set for trigger words
        ctx = {t.lower() for t in tokens if t.isalnum()}
        
        final = []
        for i, t in enumerate(tokens):
            if t.strip() and t.isalnum():
                cands = self.get_candidates(t, ctx)
                # Only use BERT if we have multiple valid candidates and the model is loaded
                if len(cands) > 1 and self.model:
                    final.append(self._match_case(t, self._rank_with_bert(tokens, i, cands)))
                else:
                    final.append(self._match_case(t, cands[0]))
            else:
                final.append(t)
        return self.post_process("".join(final))

    def normalize_slang(self, text):
        if not text: return ""
        tokens = self._tokenize(text)
        return "".join([self._match_case(t, self.slang_map.get(t.lower(), t)) for t in tokens])

    def post_process(self, text):
        if not text.strip(): return ""
        # Capitalize first valid alphanumeric char
        idx = next((i for i, c in enumerate(text) if c.isalnum()), None)
        if idx is not None: text = text[:idx] + text[idx].upper() + text[idx+1:]
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([?.!,])', r'\1', text)
        
        # Ensure ending punctuation
        if text.strip()[-1] not in ".?!": text += "."
        return text

    def _match_case(self, orig, corr):
        if orig.istitle(): return corr.capitalize()
        if orig.isupper(): return corr.upper()
        return corr

if __name__ == "__main__":
    print("="*30 + "\nUBIGKAS PROCESSOR: FINAL VERSION\n" + "="*30)
    # This will now download from Hugging Face if not cached locally
    processor = UBigkasProcessor()
    
    while True:
        try:
            txt = input("\nInput: ")
            if txt.lower() in ["exit", "quit"]: break
            print(f"Corrected: {processor.process_sentence(txt)}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break