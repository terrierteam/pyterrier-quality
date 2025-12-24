import pyterrier as pt
import pyterrier_alpha as pta


class Filter(pt.Transformer):
    def __init__(self, min_quality_score: float):
        self.min_quality_score = min_quality_score

    def transform(self, inp):
        pta.validate.document_frame(inp, extra_columns=['quality'])
        return inp[inp['quality'] >= self.min_quality_score]
