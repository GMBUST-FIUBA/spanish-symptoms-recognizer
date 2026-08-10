from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper

import spacy

class PhenotypesRecognizer():

    def __init__(self, ner_model_path=None,
                 ner_tokenizer_path=None,
                 mapper_model_path=None,
                 mapper_tokenizer_path=None,
                 ontology=None,
                 ontology_file_path=None):

        # Get NER model and tokenizer
        self.ner_model = PhenotypesDetector(model_path=ner_model_path, tokenizer_path=ner_tokenizer_path)

        # Get EL model and tokenizer
        self.mapper = PhenotypeOntologyMapper(model_path=mapper_model_path,
                                              tokenizer_path=mapper_tokenizer_path,
                                              ontology=ontology,
                                              ontology_file_path=ontology_file_path)

        # Get NLP model for sentences
        nlp = spacy.load("es_core_news_sm", disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
        nlp.add_pipe("sentencizer")
        self.text_nlp = nlp

    def recognize(self, text: str) -> list[str]:
        # Define symptoms list
        symptoms_list = []

        # Get document using SpaCy to get the text sentences
        doc = self.text_nlp(text)

        # Iterate over document sentences
        for sent in doc.sents:

            # Check if sentences are empty
            if not sent.text.strip():
                continue

            # Detect phenotypes
            sentence_results = self.ner_model.detect_phenotypes(sent.text)

            # Add elements at the end
            symptoms_list.extend(sentence_results)

        return symptoms_list

    def map(self, phenotypes_list: list[str]) -> list[str]:
        return self.mapper.map_phenotypes(phenotypes_list)

    def scan(self, text: str, only_results=True) -> list[str]:
        symptoms_list = self.recognize(text)
        hpo_codes = self.map(symptoms_list)

        return hpo_codes if only_results else {symptom : hpo_code for symptom, hpo_code in zip(symptoms_list, hpo_codes)}