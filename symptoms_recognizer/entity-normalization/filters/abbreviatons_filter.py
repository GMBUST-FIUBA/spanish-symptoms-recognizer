from normalizer_filter_base import NormalizerFilterBase

class AbbreviationsFilter(NormalizerFilterBase):
    def __init__(self):
        super().__init__()

    def apply(self, phenotypes_list):
        return []