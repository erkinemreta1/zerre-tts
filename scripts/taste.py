"""Pull the latest checkpoint from Modal and synthesize a few sentences locally.

No GPU needed: the model is 4.5M parameters, so CPU inference takes a second or
two per sentence. This avoids queueing for a GPU just to hear progress.

    uv run python scripts/taste.py                       # default sentences, both voices
    uv run python scripts/taste.py --text "Merhaba dünya." --voice MK
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEXTS = {
    "MK": "Apansız arkadaşımın bir şeyden ürktüğünü sezdim, arandım.",
    "NK": "Belki hiç ummadığımız şeyler görmemiz, beklemediğimiz zenginliklerle karşılaşmamız mümkün olur.",
}


def fetch(volume, name):
    target = ROOT / "runs" / name
    target.mkdir(parents=True, exist_ok=True)
    for remote, local in ((f"runs/{name}/inference.pt", target / "inference.pt"),
                          (f"runs/{name}/metrics.jsonl", target / "metrics.jsonl")):
        result = subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get", "--force", volume, remote, str(local)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise SystemExit(f"could not fetch {remote}: {result.stderr.strip()[:300]}")
    return target / "inference.pt"


def checkpoint_step(checkpoint):
    """The step recorded inside the checkpoint (the metrics file can be newer)."""
    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        step = state.get("step")
        return int(step) if step is not None else None
    except Exception:  # noqa: BLE001 - older checkpoints may not carry it
        return None


def checkpoint_step_metrics(metrics, step):
    if not metrics.exists():
        return {}
    lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("step") == step:
            return record
    return {}


def slug(text, limit=24):
    """Short, filesystem-safe tag so two different sentences never collide."""
    import re
    import unicodedata

    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", folded.lower())[:3]
    return "-".join(words)[:limit] or "sample"


def synthesize(checkpoint, text, voice, output):
    result = subprocess.run(
        [sys.executable, "-m", "zerre.infer", "--checkpoint", str(checkpoint), "--lang", "tr",
         "--text", text, "--voice", voice, "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise SystemExit(f"inference failed:\n{result.stderr[-600:]}")
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="zerre-tts")
    parser.add_argument("--name", default="tr-v1")
    parser.add_argument("--voice", action="append", choices=("MK", "NK"))
    parser.add_argument("--text", action="append", help="one sentence per run; voice comes from --voice")
    parser.add_argument("--output", type=Path, default=ROOT / "samples")
    args = parser.parse_args(argv)

    checkpoint = fetch(args.volume, args.name)
    metrics = ROOT / "runs" / args.name / "metrics.jsonl"
    step = checkpoint_step(checkpoint)
    record = checkpoint_step_metrics(metrics, step)

    voices = args.voice or ["MK", "NK"]
    jobs = []
    for voice in voices:
        texts = args.text or [DEFAULT_TEXTS[voice]]
        for index, text in enumerate(texts):
            jobs.append((voice, text, index))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint}")
    if step is not None:
        print(f"training step: {step}  loss {record.get('loss', float('nan')):.2f}")
    prefix = f"taste-step{step:05d}" if step is not None else "taste"
    for voice, text, index in jobs:
        output = args.output / f"{prefix}-{voice.lower()}-{slug(text)}.wav"
        detail = synthesize(checkpoint, text, voice, output)
        print(f"{output.relative_to(ROOT)}  [{voice}]  {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
