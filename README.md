# pyterrier-quality

Content quality estimation with PyTerrier.

## Models & Artifacts

The following pre-trained QualT5 models are available:

| Model ID | Base Model |
|----------|------------|
|[`pyterrier-quality/qt5-tiny`](https://huggingface.co/pyterrier-quality/qt5-tiny)|[`google/t5-efficient-tiny`](https://huggingface.co/google/t5-efficient-tiny)|

The following cached quality scores for the following datasets are also available:

| Model | Dataset | Cache |
|-------|---------|-------|
|`pyterrier-quality/qt5-tiny`|`msmarco-passage`|[`pyterrier-quality/qt5-tiny.msmarco-passage.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage.cache)|
|`pyterrier-quality/qt5-tiny`|`msmarco-passage-v2`|[`pyterrier-quality/qt5-tiny.msmarco-passage-v2.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.msmarco-passage-v2.cache)|
|`pyterrier-quality/qt5-tiny`|`cord19`|[`pyterrier-quality/qt5-tiny.cord19.cache`](https://huggingface.co/datasets/pyterrier-quality/qt5-tiny.cord19.cache)|

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
