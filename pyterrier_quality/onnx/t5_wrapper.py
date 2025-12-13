from torch import nn


class T5Wrapper(nn.Module):
    """
    Wrapper class for a T5 model to ensure compatibility with ONNX.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        return output.logits