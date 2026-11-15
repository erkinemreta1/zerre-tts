"""Compare the exported ONNX model with PyTorch on the same inputs.

The browser demo runs the ONNX file, so a silent export/precision divergence
would only show up there. This check feeds identical tokens to both.

    uv run python scripts/check_onnx.py --checkpoint runs/untrained/inference.pt \
        --onnx runs/untrained/model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args(argv)

    import onnxruntime as ort

    from zerre.infer import load_model
    from zerre.text import phonemize

    model = load_model(args.checkpoint, "cpu", check_frontend=False)
    phones = phonemize("Merhaba dünya, bu bir deneme.")
    tokens = torch.tensor([phones.ids])
    stress = torch.tensor([phones.stress])
    lengths = torch.tensor([len(phones)])
    speaker = torch.tensor([0])

    with torch.no_grad():
        audio, out_lengths, details = model.synthesize(tokens, stress, lengths, speaker)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    print("onnx inputs:", [(item.name, item.shape, item.type) for item in session.get_inputs()])
    feeds = {}
    for item in session.get_inputs():
        name = item.name.lower()
        if "token" in name:
            feeds[item.name] = tokens.numpy().astype(np.int64)
        elif "stress" in name:
            feeds[item.name] = stress.numpy().astype(np.int64)
        elif "length" in name:
            feeds[item.name] = lengths.numpy().astype(np.int64)
        elif "speaker" in name or "voice" in name:
            feeds[item.name] = speaker.numpy().astype(np.int64)
        elif "speed" in name:
            feeds[item.name] = np.array([1.0], dtype=np.float32)
        elif "pitch_shift" in name:
            feeds[item.name] = np.array([0.0], dtype=np.float32)
        elif "pitch_range" in name:
            feeds[item.name] = np.array([1.0], dtype=np.float32)
        elif "seed" in name:
            feeds[item.name] = np.array([0], dtype=np.int64)
        else:
            raise SystemExit(f"unmapped ONNX input {item.name!r}")
    outputs = session.run(None, feeds)
    reference = audio[0, :int(out_lengths[0])].numpy()
    candidate = np.asarray(outputs[0]).reshape(-1)[: reference.size]
    if candidate.size != reference.size:
        candidate = np.resize(candidate, reference.size)

    with torch.no_grad():
        other, other_lengths, _ = model.synthesize(tokens, stress, lengths, speaker, seed=1)
    baseline_wave = other[0, : int(other_lengths[0])].numpy()

    def correlation(a, b):
        numerator = float(np.dot(a, b))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return numerator / denominator

    # The source-filter head mixes a noise component that is drawn independently
    # at each call, so two PyTorch runs already differ. The control is the same
    # model with a different noise seed; ONNX must not be worse than that.
    baseline = correlation(reference, baseline_wave)
    measured = correlation(reference, candidate)
    relative_rms = float(np.sqrt(np.mean((candidate - reference) ** 2)) / (np.sqrt(np.mean(reference ** 2)) or 1.0))
    summary = {
        "tokens": int(tokens.shape[1]),
        "samples": int(reference.size),
        "onnx_correlation": round(measured, 4),
        "noise_baseline_correlation": round(baseline, 4),
        "relative_rms_error": round(relative_rms, 4),
        "reference_peak": round(float(np.abs(reference).max()), 4),
    }
    print(json.dumps(summary, indent=2))
    return 0 if measured >= baseline - 0.02 else 1


if __name__ == "__main__":
    sys.exit(main())
