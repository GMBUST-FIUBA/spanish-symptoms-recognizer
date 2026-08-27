from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import os

# Local model path
CURRENT_DIR = Path(__file__).parent.resolve()
LOCAL_MODEL_RELATIVE_PATH = "model/base-nat-data"
DEFAULT_LOCAL_MODEL_PATH = os.path.join(CURRENT_DIR, LOCAL_MODEL_RELATIVE_PATH)

class PhenotypesDetector:
    def __init__(self, model_path=None,
                 tokenizer_path=None,
                 allowed_entity_groups=None,
                 ner_model_operation_type="token-classification",
                 agg_strategy="simple"):

        # Set model and tokenizer paths
        model_path = model_path or DEFAULT_LOCAL_MODEL_PATH
        tokenizer_path = tokenizer_path or model_path

        # Get allowed entity groups
        self.allowed_entity_groups = set(allowed_entity_groups) if allowed_entity_groups else None

        # Get models
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Create pipeline
        self.ner_pipeline = pipeline(
            ner_model_operation_type,
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=agg_strategy,
        )

    def detect_phenotypes(self, sentence: str) -> list[str]:
        phenotypes_list = []

        # Use pipeline
        sentence_results = self.ner_pipeline(sentence, truncation=True, max_length=512)

        # Filter results
        for res in sentence_results:
            entity_group = res.get("entity_group")

            if not self.allowed_entity_groups or entity_group in self.allowed_entity_groups:
                phenotypes_list.append(res["word"].strip())

        return phenotypes_list