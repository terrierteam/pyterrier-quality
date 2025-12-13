import onnx
import onnxruntime as ort
from pathlib import Path
from platformdirs import user_cache_dir
import warnings


def load_onnx_model(onnx_file: Path = None,
                    model: str = 'pyterrier-quality/qt5-tiny',
                    sess_options: ort.SessionOptions = None):
    sess_options = sess_options or ort.SessionOptions()
    if onnx_file is None:
        onnx_file = create_onnx_file(model=model)
    if not onnx_file.is_file():
        export_onnx_model(model=model, onnx_file=onnx_file)

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    return ort.InferenceSession(onnx_file, sess_options=sess_options)


def export_onnx_model(model: str = 'pyterrier-quality/qt5-tiny',
                      onnx_file: Path = None):
    from pyterrier_quality import QualT5
    from pyterrier_quality.onnx.t5_wrapper import T5Wrapper
    import torch

    if onnx_file is None:
        onnx_file = create_onnx_file(model=model)

    qmodel = QualT5(model=model)

    warnings.filterwarnings("ignore")

    # Extract the necessary classes and parameters
    model = qmodel.model
    max_len = qmodel.max_len
    device = qmodel.device
    wrapped_model = T5Wrapper(model).to(device).eval()

    # Create dummy inputs
    input_ids = torch.zeros((1, max_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((1, max_len), dtype=torch.long).to(device)
    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)

    # Export the model
    torch.onnx.export(wrapped_model,
                      (input_ids, attention_mask, decoder_input_ids),
                      onnx_file,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                      output_names=["output"],
                      dynamic_axes={
                          "input_ids": {0: "batch_size", 1: "sequence_length"},
                          "attention_mask": {0: "batch_size", 1: "sequence_length"},
                          "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
                      })
    warnings.filterwarnings("default")
    print(f"ONNX model exported to {onnx_file}")


def create_onnx_file(model: str = 'pyterrier-quality/qt5-tiny'):
    cache_dir = create_cache_dir()
    model_name = model.split('/')[-1]
    return cache_dir / f"{model_name}.onnx"


def create_cache_dir() -> Path:
    cache_dir = Path(user_cache_dir(appname="pyterrier-quality"))
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created at: {cache_dir}")
    return cache_dir
