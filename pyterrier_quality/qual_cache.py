import os
import json
import numpy as np
import more_itertools
import pyterrier as pt
import pyterrier_quality
from npids import Lookup
from pyterrier_quality import Artifact


class QualCache(Artifact, pt.Indexer):
  def __init__(self, path):
    super().__init__(path)
    self._quality_scores = None
    self._docnos = None

  def index(self, it):
    return self.indexer().index(it)

  def indexer(self):
    return QualCacheIndexer(self)

  def seq_scorer(self):
    return QualCacheSeqScorer(self)

  def quality_scores(self):
    if self._quality_scores is None:
      self._quality_scores = np.memmap(os.path.join(self.path, 'quality.f8'), dtype='f8', mode='r')
    return self._quality_scores

  def docnos(self):
    if self._docnos is None:
      self._docnos = Lookup(os.path.join(self.path, 'docno.npids'))
    return self._docnos

  def quantile(self, p):
    return np.quantile(self.quality_scores(), p)

  def iter_quantiles(self):
    c = len(self.quality_scores())
    quantiles = np.argsort(np.argsort(self.quality_scores()))
    for docno, idx in zip(self.docnos(), quantiles):
      yield {'docno': docno, 'quality': idx/c}

  def get_corpus_iter(self):
    for docno, quality in zip(self.docnos(), self.quality_scores()):
      yield {'docno': docno, 'quality': quality}

  def __iter__(self):
    return self.get_corpus_iter()


class QualCacheIndexer(pt.Indexer):
  def __init__(self, cache: QualCache):
    self.cache = cache

  def index(self, it):
    if os.path.exists(self.cache.path):
      raise FileExistsError('Cache directory already exists')

    with pyterrier_quality.io.finalized_directory(self.cache.path) as d:
      count = 0
      with open(os.path.join(d, 'quality.f8'), 'wb') as fout, \
           Lookup.builder(os.path.join(d, 'docno.npids')) as docnos:
        for batch in more_itertools.chunked(it, 1000):
          batch = list(batch)
          quality_scores = np.array([d['quality'] for d in batch], dtype='f8')
          for record in batch:
            docnos.add(record['docno'])
          fout.write(quality_scores.tobytes())
          count += len(batch)
      with open(os.path.join(d, 'pt_meta.json'), 'wt') as fout:
        json.dump({
          'type': 'quality_score_cache',
          'format': 'numpy',
          'package_hint': 'pyterrier-quality',
          'count': count,
        }, fout)


class QualCacheSeqScorer(pt.Transformer):
  def __init__(self, cache: QualCache):
    self.cache = cache
    self.idx = 0

  def transform(self, inp):
    assert inp['docno'][0] == self.cache.docnos().fwd[self.idx], "detected misaligned docno when applying scores"
    quality_scores = self.cache.quality_scores()[self.idx:self.idx+len(inp)]
    self.idx += len(inp)
    return inp.assign(quality=quality_scores)
