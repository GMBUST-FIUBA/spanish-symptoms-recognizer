from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.text_parser_interface import HistoryRecordParser

class FullTextParser(HistoryRecordParser):
    def __init__(self):
        super().__init__()

    def apply(self, text: str, ner_model: PhenotypesDetector) -> list[str]:
        return ner_model.detect_phenotypes(text)