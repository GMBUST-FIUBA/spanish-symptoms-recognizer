from filters.normalizer_filter_base import NormalizerFilterBase

class PhenotypesNormalizer:
    def __init__(self, ontology):
        self.ontology = ontology

        self.filters_pipeline = self._get_filters_pipeline()

    def normalize(self, phenotypes_list: list[str]) -> list[str]:
        filter_input = phenotypes_list
        filter_output = []

        # Apply filters
        for next_filter in self.filters_pipeline:
            filter_output = next_filter.apply(filter_input)
            filter_input = filter_output

        # Return pipeline output
        return filter_output

    def _get_filters_pipeline(self) -> list[NormalizerFilterBase]:
        return []