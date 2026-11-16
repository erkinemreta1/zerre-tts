"""Write playable samples from the corpus so the target voices can be heard.

    uv run python scripts/export_samples.py --count 3 --output samples
"""

import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/tr/full.parquet"))
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument("--count", type=int, default=3, help="clips per voice")
    parser.add_argument("--min-seconds", type=float, default=3.5)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    wanted = {"MK": args.count, "NK": args.count}
    index = []
    for batch in pq.ParquetFile(args.corpus).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            voice = row["voice_id"]
            if wanted.get(voice, 0) <= 0:
                continue
            duration = row["duration_seconds"]
            if not (args.min_seconds <= duration <= args.max_seconds):
                continue
            wave, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            name = f"{voice.lower()}-{wanted[voice]}.wav"
            path = args.output / name
            sf.write(path, wave, rate, subtype="PCM_16")
            index.append({"file": name, "voice": voice, "seconds": round(len(wave) / rate, 2),
                          "text": row["text"]})
            wanted[voice] -= 1
        if not any(wanted.values()):
            break

    (args.output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in index:
        print(f"{item['file']:10s} {item['voice']} {item['seconds']:5.2f}s  {item['text'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
