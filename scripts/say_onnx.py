"""Synthesize with an ONNX file (fp32 or int8) exactly as the browser does.

    uv run python scripts/say_onnx.py --onnx runs/tr-v1/model-int8.onnx \
        --voice MK --text "Merhaba dünya." --output samples/int8-mk.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from zerre.postprocess import polish
from zerre.text import phonemize

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--voice", default="MK")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args(argv)

    import json

    import onnxruntime as ort

    model_json = args.onnx.with_name("model.json")
    if not model_json.exists():
        model_json = ROOT / "runs" / "tr-v1" / "model.json"
    config = json.loads(model_json.read_text(encoding="utf-8"))

    phones = phonemize(args.text)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    feeds = {
        "tokens": np.array([phones.ids], dtype=np.int64),
        "stress": np.array([phones.stress], dtype=np.int64),
        "voice": np.array([config["voices"].index(args.voice)], dtype=np.int64),
        "speed": np.array([args.speed], dtype=np.float32),
        "pitch_shift": np.array([0.0], dtype=np.float32),
        "pitch_range": np.array([1.0], dtype=np.float32),
    }
    audio = np.asarray(session.run(None, feeds)[0]).reshape(-1)
    if args.polish:
        audio = polish(audio, config["sample_rate"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, config["sample_rate"], subtype="PCM_16")
    print(f"{args.output}  {len(audio) / config['sample_rate']:.2f}s  peak={np.abs(audio).max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
