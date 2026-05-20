from symptoms_recognizer.ner_model.model import get_ner_model, get_tokenizer
from symptoms_recognizer.mapper.mapper import map_symptoms_to_codes

from transformers import pipeline

import spacy

class SymptomRecognizer():

    def __init__(self):
        # Get NER model and tokenizer
        self.ner_model = get_ner_model()
        self.tokenizer = get_tokenizer()
        self.ner_model_pipeline = self.__init_pipeline(self.ner_model, self.tokenizer)

        # Get NLP model for sentences
        nlp = spacy.load("es_core_news_sm", disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
        nlp.add_pipe("sentencizer")
        self.text_nlp = nlp

    def __init_pipeline(self, model, tokenizer) -> pipeline:
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            grouped_entities=True
        )

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

            # Tokenize sentence
            sentence_results = self.ner_model_pipeline(
                sent.text, 
                max_length=512, 
                truncation=True
            )

            # Store symptoms
            for results_dictionary in sentence_results:
                symptoms_list.append(results_dictionary["word"].strip())

        return symptoms_list

    def map(self, symptoms_list: list[str]) -> list[str]:
        return map_symptoms_to_codes(symptoms_list, self.tokenizer, self.ner_model)

    def scan(self, text: str, only_results=True) -> list[str]:
        symptoms_list = self.recognize(text)
        hpo_codes = self.map(symptoms_list)

        return hpo_codes if only_results else {symptom : hpo_code for symptom, hpo_code in zip(symptoms_list, hpo_codes)}