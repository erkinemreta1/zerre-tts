"""Apply the post-processing chain to wav files.

    uv run python scripts/polish.py samples/taste-step65000-mk-bugun-hava-cok.wav
    uv run python scripts/polish.py --all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from zerre.postprocess import polish

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="process every samples/taste-*.wav")
    parser.add_argument("--presence-db", type=float, default=2.5)
    parser.add_argument("--highpass", type=float, default=90.0)
    args = parser.parse_args(argv)

    files = list(args.files)
    if args.all or not files:
        files = [path for path in sorted((ROOT / "samples").glob("taste-*.wav"))
                 if "polished" not in path.name]
    if not files:
        raise SystemExit("no input files")

    for path in files:
        wave, rate = sf.read(path, dtype="float32")
        cleaned = polish(wave, rate, highpass=args.highpass, presence_gain_db=args.presence_db)
        target = path.with_name(path.stem + "-polished.wav")
        sf.write(target, cleaned, rate, subtype="PCM_16")
        try:
            shown = target.resolve().relative_to(ROOT)
        except ValueError:
            shown = target
        print(f"{shown}  peak={float(np.max(np.abs(cleaned))):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
