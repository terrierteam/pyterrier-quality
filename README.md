# pyterrier-quality

Content quality estimation with PyTerrier.

![Quality estimation pipeline](https://github.com/terrierteam/pyterrier-quality/blob/main/img/qual-predict.png)

![Filtering based on quality estimations](https://github.com/terrierteam/pyterrier-quality/blob/main/img/qual-filter.png)

## Installation

```bash
pip intall git+https://github.com/terrierteam/pyterrier-quality
```

## Models & Artifacts

The following pre-trained QualT5 models are available:

| Model ID | Base Model |
|----------|------------|
|[`pyterrier-quality/qt5-tiny`](https://huggingface.co/pyterrier-quality/qt5-tiny)|[`google/t5-efficient-tiny`](https://huggingface.co/google/t5-efficient-tiny)|
|[`pyterrier-quality/qt5-small`](https://huggingface.co/pyterrier-quality/qt5-small)|[`t5-small`](https://huggingface.co/t5-small)|
|[`pyterrier-quality/qt5-base`](https://huggingface.co/pyterrier-quality/qt5-base)|[`t5-base`](https://huggingface.co/t5-base)|

You can load these models using:

```python
from pyterrier_quality import QualT5
model = QualT5('pyterrier-quality/qt5-tiny') # or another Model ID
```

The following cached quality scores for the following datasets are also available:

| Quality Model | Dataset | Cache ID |
|---------------|---------|----------|
|`qt5-base`|`msmarco-passage`|[`pyterrier-quality/qt5-base.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-base.msmarco-passage.cache)|
|`qt5-small`|`msmarco-passage`|[`pyterrier-quality/qt5-small.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-small.msmarco-passage.cache)|
|`qt5-tiny`|`msmarco-passage`|[`pyterrier-quality/qt5-tiny.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage.cache)|
|*(random)*|`msmarco-passage`|[`pyterrier-quality/rand.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/rand.msmarco-passage.cache)|
|ITN|`msmarco-passage`|[`pyterrier-quality/itn.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/itn.msmarco-passage.cache)|
|CDD|`msmarco-passage`|[`pyterrier-quality/cdd.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/cdd.msmarco-passage.cache)|
|EPIC (Qual.)|`msmarco-passage`|[`pyterrier-quality/epic.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/epic.msmarco-passage.cache)|
|TAS-B (Mag.)|`msmarco-passage`|[`pyterrier-quality/tasb-mag.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/tasb-mag.msmarco-passage.cache)|
|t5-base (Ppl.)|`msmarco-passage`|[`pyterrier-quality/t5-ppl.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/t5-ppl.msmarco-passage.cache)|
|gpt2 (Ppl.)|`msmarco-passage`|[`pyterrier-quality/gpt2-ppl.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/gpt2-ppl.msmarco-passage.cache)|
|`qt5-tiny`|`cord19`|[`pyterrier-quality/qt5-tiny.cord19.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.cord19.cache)|
|*(random)*|`cord19`|[`pyterrier-quality/rand.cord19.cache`](https://huggingface.co/datasets/pyterrier-quality/rand.cord19.cache)|
|`qt5-tiny`|`msmarco-passage-v2`|[`pyterrier-quality/qt5-tiny.msmarco-passage-v2.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage-v2.cache)|
|*(random)*|`msmarco-passage-v2`|[`pyterrier-quality/rand.msmarco-passage-v2.cache`](https://huggingface.co/datasets/pyterrier-quality/rand.msmarco-passage-v2.cache)|

You can load a cache using:

```python
from pyterrier_quality import QualCache
cache = QualCache.from_url('hf:pyterrier-quality/qt5-tiny.msmarco-passage.cache') # or another Cache ID (note the hf: prefix)
```

For convenience, specifying the `@quantiles` branch on any of the caches provides a version of the quality scores
converted into the corresponding quantile score. For example:

```python
from pyterrier_quality import QualCache
cache = QualCache.from_url('hf:pyterrier-quality/qt5-tiny.msmarco-passage.cache@quantiles')
```

The following indexes are available, based on the quality scores above:

| Quality Model | Dataset | PISA (BM25) | PISA (SPLADE (lg)) | FLEX (TAS-B) |
|---------------|---------|-------------|--------------------|--------------|
|`qt5-tiny`|`msmarco-passage`|[`qt5-tiny.msmarco-passage.pisa`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage.pisa)|[`qt5-tiny.msmarco-passage.splade-lg.pisa`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage.splade-lg.pisa)|
|*(random)*|`msmarco-passage`|[`rand.msmarco-passage.pisa`](https://huggingface.co/datasets/pyterrier-quality/rand.msmarco-passage.pisa)|
|`qt5-tiny`|`cord19`|[`qt5-tiny.cord19.pisa`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.cord19.pisa)|[`qt5-tiny.cord19.splade-lg.pisa`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.cord19.splade-lg.pisa)|
|*(random)*|`cord19`|[`rand.cord19.pisa`](https://huggingface.co/datasets/pyterrier-quality/rand.cord19.pisa)|
|`qt5-tiny`|`msmarco-passage-v2`|[`qt5-tiny.msmarco-passage-v2.pisa`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage-v2.pisa)|[`qt5-tiny.msmarco-passage-v2.splade-lg.pisa`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage-v2.splade-lg.pisa)|
|*(random)*|`msmarco-passage-v2`|[`rand.msmarco-passage-v2.pisa`](https://huggingface.co/datasets/pyterrier-quality/rand.msmarco-passage-v2.pisa)|


# Citation

This repository is for the paper **Neural Passage Quality Estimation for Static Pruning** at SIGIR 2024.
If you use this work, please cite:

```bibtex
@inproceedings{DBLP:conf/sigir/ChangMMM24,
  author       = {Xuejun Chang and
                  Debabrata Mishra and
                  Craig Macdonald and
                  Sean MacAvaney},
  title        = {Neural Passage Quality Estimation for Static Pruning},
  booktitle    = {Proceedings of the 47th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2024},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3626772.3657765},
  doi          = {10.1145/3626772.3657765}
}
```
