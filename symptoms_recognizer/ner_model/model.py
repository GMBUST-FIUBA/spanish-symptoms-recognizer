from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import os

# Local model path
CURRENT_DIR = Path(__file__).parent.resolve()
LOCAL_MODEL_RELATIVE_PATH = "model/base-nat-data"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)

class PhenotypesDetector:
    def __init__(self, model_path=None, tokenizer_path=None):
        # Get NER model
        if model_path:
            self.model = self._get_ner_model(model_path)
        else:
            self.model = self._get_ner_model(DEFAULT_LOCAL_MODEL_PATH)

        # Get tokenizer
        if tokenizer_path:
            self.tokenizer = self._get_ner_tokenizer(tokenizer_path)
        else:
            self.tokenizer = self._get_ner_tokenizer(DEFAULT_LOCAL_MODEL_PATH)

        # Create pipeline
        self.ner_model_pipeline = self.__init_pipeline(self.model, self.tokenizer)

    def _get_ner_tokenizer(self, path):
        return AutoTokenizer.from_pretrained(path)

    def _get_ner_model(self, path):
        return AutoModelForTokenClassification.from_pretrained(path)

    def __init_pipeline(self, model, tokenizer) -> pipeline:
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    def detect_phenotypes(self, sentence):
        phenotypes_list = []

        # Tokenize and decode after to force truncation and padding
        tokens = self.tokenizer.encode(
            sentence, 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )
        truncated_sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)

        # Detect phenotypes
        sentence_results = self.ner_model_pipeline(truncated_sentence)

        # Get solutions
        for results_dictionary in sentence_results:
            phenotypes_list.append(results_dictionary["word"].strip())

        return phenotypes_list