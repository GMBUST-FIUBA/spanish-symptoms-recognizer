from abc import ABC, abstractmethod

from symptoms_recognizer.ner_model.model import PhenotypesDetector

class HistoryRecordParser(ABC):
    @abstractmethod
    def apply(self, text: str, ner_model: PhenotypesDetector) -> list[str]:
        pass