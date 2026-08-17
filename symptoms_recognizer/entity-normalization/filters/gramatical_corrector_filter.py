from normalizer_filter_base import NormalizerFilterBase

class GramaticalCorrectorFilter(NormalizerFilterBase):
    def __init__(self):
        super().__init__()

    def apply(self, phenotypes_list):
        return []