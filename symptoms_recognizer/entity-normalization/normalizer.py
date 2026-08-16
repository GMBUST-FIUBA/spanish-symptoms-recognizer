class PhenotypesNormalizer:
    def __init__(self, ontology):
        self.ontology = ontology

    def normalize(self, phenotypes_list: list[str]) -> list[str]:
        # Correct misspellings
        correctly_spelled_phenotypes = self._correct_misspelings(phenotypes_list)

        # Convert to ontology and return
        return self._normalize_phenotypes_to_ontology(correctly_spelled_phenotypes)

    def _correct_misspelings(self, phenotypes_list: list[str]) -> list[str]:
        return []

    def _normalize_phenotypes_to_ontology(self, phenotypes_list) -> list[str]:
        return []