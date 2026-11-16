"""Compare fp32 and int8 ONNX on held-out clips: waveform agreement and WER.

    uv run python scripts/compare_onnx_wer.py --clips 8
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def wer(reference, hypothesis):
    import jiwer

    return float(jiwer.wer(reference, hypothesis))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--asr", default="large-v3")
    parser.add_argument("--fp32", type=Path, default=ROOT / "runs/tr-v1/model.onnx")
    parser.add_argument("--int8", type=Path, default=ROOT / "runs/tr-v1/model-int8.onnx")
    args = parser.parse_args(argv)

    from faster_whisper import WhisperModel

    rows = [json.loads(line) for line in
            (ROOT / "data/tr/annotations/test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.clips]
    asr = WhisperModel(args.asr, device="cpu", compute_type="int8", cpu_threads=8)

    def transcribe(path):
        audio, rate = sf.read(path, dtype="float32")
        segments, _ = asr.transcribe(audio, language="tr", beam_size=5, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    scores = {"fp32": [], "int8": []}
    correlations = []
    with tempfile.TemporaryDirectory() as directory:
        for index, row in enumerate(rows):
            outputs = {}
            for name, model in (("fp32", args.fp32), ("int8", args.int8)):
                path = Path(directory) / f"{index}-{name}.wav"
                subprocess.run([sys.executable, "scripts/say_onnx.py", "--onnx", str(model),
                                "--voice", row["speaker"], "--text", row["text"],
                                "--output", str(path), "--polish"],
                               cwd=ROOT, check=True, capture_output=True)
                outputs[name] = path
                hypothesis = transcribe(path)
                scores[name].append(wer(row["text"], hypothesis))
            a, _ = sf.read(outputs["fp32"], dtype="float32")
            b, _ = sf.read(outputs["int8"], dtype="float32")
            length = min(len(a), len(b))
            if length:
                denom = float(np.linalg.norm(a[:length]) * np.linalg.norm(b[:length])) or 1.0
                correlations.append(float(np.dot(a[:length], b[:length])) / denom)

    summary = {
        "clips": len(rows),
        "wer_fp32": round(float(np.mean(scores["fp32"])), 4),
        "wer_int8": round(float(np.mean(scores["int8"])), 4),
        "mean_correlation": round(float(np.mean(correlations)), 4) if correlations else None,
        "sizes_mb": {"fp32": round(args.fp32.stat().st_size / 1e6, 2),
                     "int8": round(args.int8.stat().st_size / 1e6, 2)},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
