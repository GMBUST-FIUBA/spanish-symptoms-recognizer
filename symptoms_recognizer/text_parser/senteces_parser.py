from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.text_parser_interface import HistoryRecordParser

import spacy

class SentencesParser(HistoryRecordParser):
    def __init__(self):
        self.text_nlp = spacy.blank("es")
        self.text_nlp.add_pipe("sentencizer")

    def apply(self, text: str, ner_model: PhenotypesDetector):
        phenotypes_list = []
        doc = self.text_nlp(text)

        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            if not sentence_text:
                continue

            # Detect phenotypes by sentence
            sentence_results = ner_model.detect_phenotypes(sentence_text)
            phenotypes_list.extend(sentence_results)

        return phenotypes_list