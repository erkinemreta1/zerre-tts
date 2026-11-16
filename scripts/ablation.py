"""Four-way ablation: fp32 vs int8, raw vs post-processed.

Synthesizes the same sentences with each combination, measures WER on held-out
clips, the low-band energy share and the file size, writes one showcase wav per
combination and prints a table.

    uv run python scripts/ablation.py --clips 8
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
SHOWCASE = "Bugün hava çok güzel, sahilde yürüyüş yapalım mı?"

VARIANTS = {
    "fp32-ham": (ROOT / "runs/tr-v1/model.onnx", False),
    "fp32-filtreli": (ROOT / "runs/tr-v1/model.onnx", True),
    "int8-ham": (ROOT / "runs/tr-v1/model-int8.onnx", False),
    "int8-filtreli": (ROOT / "runs/tr-v1/model-int8.onnx", True),
}


def low_band_share(wave, rate, cutoff=120.0):
    if wave.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) ** 2
    freqs = np.fft.rfftfreq(len(wave), 1 / rate)
    total = spectrum.sum() or 1.0
    return float(spectrum[freqs < cutoff].sum() / total)


def synth(model, text, voice, output, polish):
    command = [sys.executable, "scripts/say_onnx.py", "--onnx", str(model), "--voice", voice,
               "--text", text, "--output", str(output)]
    if polish:
        command.append("--polish")
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--asr", default="large-v3")
    args = parser.parse_args(argv)

    import jiwer
    from faster_whisper import WhisperModel

    rows = [json.loads(line) for line in
            (ROOT / "data/tr/annotations/test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.clips]
    asr = WhisperModel(args.asr, device="cpu", compute_type="int8", cpu_threads=8)

    def transcribe(path):
        audio, rate = sf.read(path, dtype="float32")
        segments, _ = asr.transcribe(audio, language="tr", beam_size=5, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, (model, polished) in VARIANTS.items():
            scores, shares, seconds = [], [], []
            for index, row in enumerate(rows):
                path = Path(directory) / f"{name}-{index}.wav"
                synth(model, row["text"], row["speaker"], path, polished)
                wave, rate = sf.read(path, dtype="float32")
                shares.append(low_band_share(wave, rate))
                seconds.append(len(wave) / rate)
                scores.append(float(jiwer.wer(row["text"], transcribe(path))))
            results[name] = {
                "wer": round(float(np.mean(scores)), 4),
                "low_band_pct": round(float(np.mean(shares)) * 100, 1),
                "seconds": round(float(np.mean(seconds)), 2),
                "size_mb": round(model.stat().st_size / 1e6, 2),
            }

        showcase = ROOT / "samples"
        showcase.mkdir(exist_ok=True)
        for name, (model, polished) in VARIANTS.items():
            output = showcase / f"ablation-{name}.wav"
            synth(model, SHOWCASE, "MK", output, polished)
            print(f"wrote {output.relative_to(ROOT)}")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    (ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "reports" / "ablation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
