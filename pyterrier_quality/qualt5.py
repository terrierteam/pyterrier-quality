import itertools
import pyterrier as pt
import torch
from torch.nn import functional as F
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

import pyterrier as pt


class QualT5(pt.Transformer):
    def __init__(self,
                 model='pyterrier-quality/qt5-tiny',
                 *,
                 batch_size=100,
                 verbose=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model, fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        targets = ['true', 'false'] if not hasattr(self.model.config, 'targets') else self.model.config.targets
        # replace the lm_head with a version that only inclues the "true" and "false" tokens to save compute
        self.model.lm_head.weight = torch.nn.Parameter(self.model.lm_head.weight[[
            self.tokenizer.encode(targets[0])[0],
            self.tokenizer.encode(targets[1])[0],
        ]])
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, inp):
        assert 'text' in inp.columns
        texts = inp['text'].to_list()

        scores = []
        it = range(0, len(texts), self.batch_size)
        dec_ids = torch.full(
            (self.batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        if self.verbose:
            it = pt.tqdm(it, desc=self.model_name, unit='batches')

        for start_idx in it:
            rng = slice(start_idx, start_idx + self.batch_size)
            enc = self.tokenizer.batch_encode_plus([f'Document: {d} Relevant:' for d in texts[rng]], return_tensors='pt', padding='longest', max_length=512, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            enc['decoder_input_ids'] = dec_ids[:len(texts[rng])]
            if scores:
              scores[-1] = scores[-1].cpu().detach().tolist()
            with torch.no_grad(), torch.autocast(device_type=self.device.type):
                result = self.model(**enc).logits
                result = result[:, 0]
                scores.append(F.log_softmax(result, dim=1)[:, 0])
        if scores:
          scores[-1] = scores[-1].cpu().detach().tolist()
        scores = list(itertools.chain.from_iterable(scores))

        return inp.assign(quality=scores)
