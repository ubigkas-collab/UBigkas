import torch
import re
import os
import sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==========================================
# 1. CONFIGURATION
# ==========================================
# DIRECT LINK TO HUGGING FACE MODEL
MODEL_PATH = "Vinci14/tagalog_ner_model"
CONFIDENCE_THRESHOLD = 0.85 

# MASTER DICTIONARY (Finalized with Rural, Medical, and Modern contexts)
VERB_TYPES = {
    # MAG verbs
    "luto": "MAG", "laro": "MAG", "tanim": "MAG", "aral": "MAG", 
    "linis": "MAG", "trabaho": "MAG", "dilig": "MAG", "pitas": "MAG",
    "nood": "MAG", "pasa": "MAG", "type": "MAG", "tago": "MAG",
    "gamot": "MAG", "bilang": "MAG", "pirma": "MAG", "punas": "MAG",
    "parada": "MAG", "pasyal": "MAG", "pinta": "MAG", "tugtog": "MAG",
    "kanta": "MAG", "film": "MAG", "ensayo": "MAG", "pakain": "MAG",
    "celebrate": "MAG", "report": "MAG", "pila": "MAG",
    # UM verbs
    "kain": "UM", "inom": "UM", "punta": "UM", "bili": "UM", 
    "alis": "UM", "iyak": "UM", "sakay": "UM", "takbo": "UM",
    "uwi": "UM", "dating": "UM", "pasok": "UM", "labas": "UM",
    "kuha": "UM", "tawak": "UM", "baba": "UM", "ani": "UM",
    "simba": "UM", "ihip": "UM", "sikat": "UM", "lubog": "UM", "yanig": "UM",
    # IN verbs
    "basa": "IN", "sulat": "IN", "gamit": "IN", "dala": "IN", "huli": "IN", "buhos": "IN",
    # AN verbs
    "hugas": "AN", "bukas": "AN", "sarado": "AN", "bayad": "AN", "bigay": "AN"
}

PAST_KEYWORDS = {"kahapon", "kanina", "noon", "kagabi", "nakaraan", "dati", "noong"}
FUTURE_KEYWORDS = {"bukas", "mamaya", "susunod", "balang_araw", "sa"}
PRESENT_KEYWORDS = {"ngayon", "kasalukuyan", "palagi", "tuwing", "habang", "gabi-gabi", "araw-araw"}
TIME_ADVERBS = PAST_KEYWORDS | FUTURE_KEYWORDS | PRESENT_KEYWORDS

# ==========================================
# 2. DYNAMIC CONJUGATION ENGINE
# ==========================================
def get_redup(root):
    if not root: return ""
    if root[0] in "aeiou": return root[0]
    match = re.match(r"([^aeiou]+[aeiou])", root)
    return match.group(1) if match else root[0]

def insert_infix(root, infix):
    if not root: return root
    if root[0] in "aeiou": return f"{infix}{root}"
    match = re.match(r"([^aeiou]+)(.*)", root)
    if match: return f"{match.group(1)}{infix}{match.group(2)}"
    return f"{infix}{root}"

def conjugate(root, v_type, tense):
    # Irregular Overrides
    if root == "nood":
        if tense == "past": return "nanood"
        if tense == "present": return "nanonood"
        return "manonood"

    if root == "bukas" and v_type == "AN":
        redup = get_redup(root)
        if tense == "future": return f"{redup}buksan"
        if tense == "past": return "binuksan"
        return f"bi{redup}buksan"
        
    redup = get_redup(root)
    if v_type == "MAG":
        sep = "-" if root[0] in "aeiou" else ""
        if tense == "future": return f"mag{sep}{redup}{root}"
        if tense == "present": return f"nag{sep}{redup}{root}"
        return f"nag{sep}{root}"
    elif v_type == "UM":
        if tense == "future": return f"{redup}{root}"
        if tense == "past": return insert_infix(root, "um")
        return insert_infix(f"{redup}{root}", "um")
    elif v_type == "IN":
        suffix = "hin" if root[-1] in "aeiou" else "in"
        if tense == "future": return f"{redup}{root}{suffix}"
        if tense == "past": return insert_infix(root, "in")
        return insert_infix(f"{redup}{root}", "in")
    elif v_type == "AN":
        suffix = "han" if root[-1] in "aeiou" else "an"
        if tense == "future": return f"{redup}{root}{suffix}"
        if tense == "past": return f"{insert_infix(root, 'in')}{suffix}"
        return f"{insert_infix(f'{redup}{root}', 'in')}{suffix}"
    return root

# ==========================================
# 3. PREDICTION & RECONSTRUCTION
# ==========================================
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")
        try:
            # use_fast=True is important for correct word_id mapping
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

def predict_tags(sentence):
    load_model()
    tokens = sentence.split()
    
    # Tokenize with offset mapping to align subwords to original words
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    
    confidences, pred_ids = torch.max(probs, dim=-1)
    pred_ids = pred_ids[0].tolist()
    confidences = confidences[0].tolist()
    word_ids = inputs.word_ids(batch_index=0)
    
    tags = []
    scores = []
    used_word = set()
    
    # Iterate through tokens and pick the label of the first sub-token for each word
    for pid, score, wid in zip(pred_ids, confidences, word_ids):
        if wid is not None and wid not in used_word:
            tags.append(model.config.id2label[pid])
            scores.append(score)
            used_word.add(wid)
            
    return tokens, tags, scores

def insert_markers(tokens, tags, scores):
    out = []
    tense = "base"
    token_set = set(t.lower() for t in tokens)
    
    # 1. Detect Tense from Context Keywords
    if any(w in token_set for w in FUTURE_KEYWORDS): tense = "future"
    elif any(w in token_set for w in PAST_KEYWORDS): tense = "past"
    elif any(w in token_set for w in PRESENT_KEYWORDS): tense = "present"
    
    # 2. Fallback to NER Tags for Tense if Context is Missing
    if tense == "base":
        if "B-FUTURE_ADV" in tags: tense = "future"
        elif "B-PRESENT_ADV" in tags: tense = "present"
        elif "B-PAST_ADV" in tags: tense = "past"
    
    # 3. Reconstruction Loop
    for i, (tok, tag, score) in enumerate(zip(tokens, tags, scores)):
        # Filter low confidence predictions
        if score < CONFIDENCE_THRESHOLD: tag = "O"
        
        prev_word = out[-1].lower() if out else ""
        
        # Skip existing markers in input to avoid duplication
        if tok.lower() in ["ang", "ng", "sa", "ay", "mga", "si", "ni", "nasa"]:
            out.append(tok)
            continue

        is_candidate_verb = tok.lower() in VERB_TYPES
        is_time_usage = (tok.lower() in TIME_ADVERBS) and tense != "base"
        
        # Handle Verbs
        if is_candidate_verb and not is_time_usage:
            word_to_add = conjugate(tok.lower(), VERB_TYPES[tok.lower()], tense)
        else:
            word_to_add = tok

        # Handle Marker Insertion based on Tags
        if tag == "B-AY":
            if prev_word != "ay" and i > 0: out.append("ay")
            out.append(word_to_add)
        elif tag == "B-SA":
            if word_to_add == "sa" or prev_word == "sa" or prev_word == "nasa":
                out.append(word_to_add)
            else:
                if prev_word not in ["sa", "ng", "ang", "nasa"]: out.append("sa")
                out.append(word_to_add)
        elif tag == "B-NG":
            if word_to_add == "ng" or prev_word in ["ng", "ay", "ang"]:
                out.append(word_to_add)
            else:
                out.append("ng")
                out.append(word_to_add)
        else:
            out.append(word_to_add)
            
    # 4. Final Cleanup (De-duplication and Preposition Fixing)
    res = " ".join(out)
    res = re.sub(r'\b(ng|sa|ay)\s+\1\b', r'\1', res) 
    positions = r"likod|harap|taas|baba|loob|labas|gilid|gitna|ibabaw|ilalim"
    # Fix "sa ilalim mesa" -> "sa ilalim ng mesa"
    res = re.sub(rf"(?:\b(?:sa|ng)\b\s*)*\b({positions})\b(?:\s+\b(?:sa|ng)\b)*\s+(\w+)", r"sa \1 ng \2", res)
    return res

if __name__ == "__main__":
    print("-" * 50 + "\nUBIGKAS ENGINE ACTIVE\n" + "-" * 50)
    print(f"Target Model: {MODEL_PATH}")
    while True:
        try:
            sentence = input("\nInput:  ").strip()
            if sentence.lower() in ["exit", "q"]: break
            if not sentence: continue
            
            tokens, tags, scores = predict_tags(sentence)
            
            # Debug view (optional - helps verify if tags are correct)
            # print(f"DEBUG Tags: {list(zip(tokens, tags))}")
            
            print(f"Output: {insert_markers(tokens, tags, scores)}")
        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Error processing input: {e}")