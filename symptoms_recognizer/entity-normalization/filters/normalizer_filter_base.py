from abc import ABC, abstractmethod

class NormalizerFilterBase(ABC):
    @abstractmethod
    def apply(self, phenotypes_list: list[str]) -> list[str]:
        pass