from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper

import spacy

import spacy
from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper

class PhenotypesRecognizer:
    def __init__(
        self, 
        ner_model_path=None,
        ner_tokenizer_path=None,
        mapper_model_path=None,
        mapper_tokenizer_path=None,
        ontology=None,
        ontology_file_path=None,
        allowed_entity_groups=None,
        agg_strategy="simple"
    ):

        # Initialize NER model
        self.ner_model = PhenotypesDetector(
            model_path=ner_model_path, 
            tokenizer_path=ner_tokenizer_path,
            allowed_entity_groups=allowed_entity_groups,
            agg_strategy=agg_strategy
        )

        # Initialize mapper
        self.mapper = PhenotypeOntologyMapper(
            model_path=mapper_model_path,
            tokenizer_path=mapper_tokenizer_path,
            ontology=ontology,
            ontology_file_path=ontology_file_path
        )

        # Initialize text splitter
        self.text_nlp = spacy.blank("es")
        self.text_nlp.add_pipe("sentencizer")

    def recognize(self, text: str) -> list[str]:
        symptoms_list = []
        doc = self.text_nlp(text)

        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            if not sentence_text:
                continue

            # Detect phenotypes by sentence
            sentence_results = self.ner_model.detect_phenotypes(sentence_text)
            symptoms_list.extend(sentence_results)

        return symptoms_list

    def map(self, phenotypes_list: list[str]) -> list[str]:
        return self.mapper.map_phenotypes(phenotypes_list)

    def scan(self, text: str, only_results=False) -> list:
        symptoms_list = self.recognize(text)
        hpo_codes = self.map(symptoms_list)

        if only_results:
            return hpo_codes

        return [(symptom, hpo_code) for symptom, hpo_code in zip(symptoms_list, hpo_codes) if hpo_code != "None"]