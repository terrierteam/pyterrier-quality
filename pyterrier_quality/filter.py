import pyterrier as pt


class Filter(pt.Transformer):
    def __init__(self, min_quality_score: float):
        self.min_quality_score = min_quality_score

    def transform(self, inp):
        assert 'quality' in inp
        return inp[inp['quality'] >= min_quality_score]
