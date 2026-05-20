from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

import os

# Local model path
CURRENT_DIR = Path(__file__).parent.resolve()
LOCAL_MODEL_RELATIVE_PATH = "model/output"
absolute_model_path = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)

def get_tokenizer():
    return AutoTokenizer.from_pretrained(absolute_model_path)

def get_ner_model():
    return AutoModelForTokenClassification.from_pretrained(absolute_model_path)