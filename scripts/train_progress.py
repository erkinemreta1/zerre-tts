"""Print the training progress from the local copy of the metrics journal.

    uv run python scripts/train_progress.py --name tr-v1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="tr-v1")
    parser.add_argument("--total", type=int, default=65000)
    parser.add_argument("--fetch", action="store_true", help="pull the journal from the Modal volume first")
    args = parser.parse_args(argv)

    journal = ROOT / "runs" / args.name / "metrics.jsonl"
    if args.fetch:
        import subprocess

        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "zerre-tts",
                        f"runs/{args.name}/metrics.jsonl", str(journal)],
                       check=True, capture_output=True)
    if not journal.exists():
        raise SystemExit(f"{journal} is missing; run with --fetch")

    lines = [line for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("the journal is empty")
    last = None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in record:
            last = record
            break
    if last is None:
        raise SystemExit("no step records in the journal")

    step = last["step"]
    speed = last.get("steps_per_second") or 0
    remaining = max(args.total - step, 0)
    hours = remaining / speed / 3600 if speed else 0
    print(f"step      {step:,} / {args.total:,}  ({step / args.total * 100:.0f}%)")
    print(f"loss      {last.get('loss', float('nan')):.2f}")
    print(f"speed     {speed:.2f} step/s ({last.get('audio_seconds_per_second', 0):.0f}x audio)")
    print(f"remaining {remaining:,} steps ≈ {hours:.1f} h")
    validation = {key: value for key, value in last.items() if key.startswith("validation_")}
    if validation:
        print("validation " + ", ".join(f"{key.replace('validation_', '')}={value:.3f}"
                                        for key, value in list(validation.items())[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
