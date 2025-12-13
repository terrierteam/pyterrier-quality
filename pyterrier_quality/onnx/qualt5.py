import itertools
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig

from pyterrier_quality.onnx.utils import load_onnx_model


class ONNXQualT5:
    def __init__(self,
                 model: str = "pyterrier-quality/qt5-tiny",
                 onnx_file: Path = None,
                 batch_size: int = 100,
                 max_len: int = 512,
                 prompt: str = "Document: {} Relevant:",
                 verbose: bool = False,
                 sess_options: ort.SessionOptions = None,
                 ):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, fast=True)
        self.config = AutoConfig.from_pretrained(model)

        sess_options = sess_options or ort.SessionOptions()
        self.ort_sess = load_onnx_model(model=model, onnx_file=onnx_file, sess_options=sess_options)

        self.batch_size = batch_size
        self.verbose = verbose
        self.max_len = max_len
        self.prompt = prompt

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert 'text' in inp.columns
        texts = inp['text'].to_list()

        scores = []
        it = range(0, len(texts), self.batch_size)
        dec_ids = np.full(
            (self.batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=np.int64,
        )
        if self.verbose:
            it = tqdm(it, desc=self.model_name, unit='batches')

        for start_idx in it:
            rng = slice(start_idx, start_idx + self.batch_size)
            enc = self.tokenizer.batch_encode_plus([self.prompt.format(d) for d in texts[rng]], return_tensors='np',
                                                   padding='longest', max_length=self.max_len, truncation=True)
            enc['decoder_input_ids'] = dec_ids[:len(texts[rng])]
            enc = {k: np.ascontiguousarray(v) for k, v in enc.items()}
            if scores:
                scores[-1] = scores[-1].tolist()

            result = self.ort_sess.run(None, enc)[0][:, 0]
            scores.append(log_softmax(x=result, dim=1)[:, 0])

        if scores:
            scores[-1] = scores[-1].tolist()
        scores = list(itertools.chain.from_iterable(scores))
        return inp.assign(quality=scores)


def log_softmax(x, dim=1):
    x_max = np.max(x, axis=dim, keepdims=True)
    shifted_x = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted_x), axis=dim, keepdims=True))
    return shifted_x - log_sum_exp