import os
import json
import numpy as np
import more_itertools
import pyterrier as pt
import pyterrier_quality
from npids import Lookup
from pyterrier_quality import Artifact


class ScoreCache(Artifact, pt.Indexer):
  def __init__(self, path):
    super().__init__(path)
    self._mmap = None
    self._docnos = None

  def index(self, it):
    return self.indexer().index(it)

  def indexer(self):
    return ScoreCacheIndexer(self)

  def seq_scorer(self):
    return ScoreCacheSeqScorer(self)

  def mmap(self):
    if self._mmap is None:
      self._mmap = np.memmap(os.path.join(self.path, 'scores.f8'), dtype='f8', mode='r')
    return self._mmap

  def docnos(self):
    if self._docnos is None:
      self._docnos = Lookup(os.path.join(self.path, 'docnos.npids'))
    return self._docnos

  def quantile(self, p):
    return np.quantile(self.mmap(), p)


class ScoreCacheIndexer(pt.Indexer):
  def __init__(self, cache: ScoreCache):
    self.cache = cache

  def index(self, it):
    if os.path.exists(self.cache.path):
      raise FileExistsError('Cache directory already exists')

    with pyterrier_quality.io.finalized_directory(self.cache.path) as d:
      count = 0
      with open(os.path.join(d, 'scores.f8'), 'wb') as fout, \
           Lookup.builder(os.path.join(d, 'docnos.npids')) as docnos:
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


class ScoreCacheSeqScorer(pt.Transformer):
  def __init__(self, cache: ScoreCache):
    self.cache = cache
    self.idx = 0

  def transform(self, inp):
    assert inp['docno'][0] == self.cache.docnos().fwd[self.idx], "detected misaligned docno when applying scores"
    quality_scores = self.cache.mmap()[self.idx:self.idx+len(inp)]
    self.idx += len(inp)
    return inp.assign(quality=quality_scores)
