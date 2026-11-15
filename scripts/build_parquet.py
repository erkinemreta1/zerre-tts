"""Build the zerre-tts training parquet from the Turkish source datasets.

The sources are two 48 kHz audiobook corpora on Hugging Face. This script
downloads shards, decodes the embedded audio, resamples to 24 kHz (the acoustic
config of the model), filters and de-duplicates, then writes a single parquet
whose columns match what ``zerre.prepare`` expects:
``id, text_id, voice_id, text, split, category, origin, source_id,
source_license, source_url, duration_seconds, sample_rate, asr_transcript, audio``.

Splitting is by text hash, so the same sentence never appears in two splits.

    uv run python scripts/build_parquet.py --shards 1 --output data/tr/pilot.parquet
"""

import argparse
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

DATASETS = {
    "MK": ("omersaidd/tts_mazlum_kiper_tur", "Mazlum Kiper", "mit"),
    "NK": ("omersaidd/tts_nisan_kumru_tur", "Nisan Kumru", "mit"),
}

TARGET_RATE = 24000
MIN_DURATION = 1.0
MAX_DURATION = 15.0
MIN_CHARS = 15
MAX_CHARS = 300
TRAIN_BUCKETS = 980  # 0..979 train, 980..989 validation, 990..999 test

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bucket(text):
    digest = hashlib.sha256(normalize(text).casefold().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def split_of(text):
    value = bucket(text)
    if value < TRAIN_BUCKETS:
        return "train"
    return "validation" if value < TRAIN_BUCKETS + 10 else "test"


def decode_and_resample(audio):
    import soundfile as sf
    import soxr

    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    wave, rate = sf.read(source, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != TARGET_RATE:
        wave = soxr.resample(wave, rate, TARGET_RATE, quality="VHQ")
    return wave


def encode(wave):
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wave, TARGET_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def accept(wave, text):
    """Cheap quality gate. Transcript correctness is a separate pass."""
    import numpy as np

    if not (MIN_DURATION <= len(wave) / TARGET_RATE <= MAX_DURATION):
        return "duration"
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return "text length"
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak < 0.01:
        return "silent"
    if peak > 0.999:
        return "clipped"
    rms = float(np.sqrt(np.mean(wave ** 2)))
    if rms < 1e-4:
        return "silent"
    return None


def shards_of(repo, limit):
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, files_metadata=True)
    files = sorted(
        (sibling for sibling in info.siblings if sibling.rfilename.endswith(".parquet")),
        key=lambda sibling: sibling.rfilename,
    )
    return files[:limit] if limit else files


def run(voices, shards, max_hours, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    # Each voice gets an equal share of the budget, so one book cannot eat it all.
    per_voice_cap = max_hours * 3600 / max(len(voices), 1)
    total_seconds = 0.0
    completed_seconds = 0.0
    stats = {"seen": 0, "kept": 0, "rejected": {}, "per_voice": {}}
    seen_texts = set()

    schema = pa.schema([
        ("id", pa.string()), ("text_id", pa.string()), ("voice_id", pa.string()),
        ("text", pa.string()), ("split", pa.string()), ("category", pa.string()),
        ("origin", pa.string()), ("source_id", pa.string()), ("source_license", pa.string()),
        ("source_url", pa.string()), ("duration_seconds", pa.float64()),
        ("sample_rate", pa.int64()), ("asr_transcript", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, schema)
    try:
        for voice, (repo, name, licence) in DATASETS.items():
            if voice not in voices:
                continue
            kept_voice = 0
            seconds_voice = 0.0
            for shard in shards_of(repo, shards):
                print(f"{voice}: {shard.rfilename}", flush=True)
                local = hf_hub_download(repo, shard.rfilename, repo_type="dataset")
                parquet = pq.ParquetFile(local)
                rows, batch_keys = [], []
                for batch in parquet.iter_batches(batch_size=16):
                    for row in batch.to_pylist():
                        stats["seen"] += 1
                        text = normalize(row.get("text") or "")
                        key = hashlib.sha256(text.casefold().encode()).hexdigest()
                        if f"{voice}:{key}" in seen_texts:
                            stats["rejected"]["duplicate text"] = stats["rejected"].get("duplicate text", 0) + 1
                            continue
                        try:
                            wave = decode_and_resample(row["audio"])
                        except Exception as error:  # noqa: BLE001 - count and continue
                            reason = f"decode: {type(error).__name__}"
                            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                            continue
                        reason = accept(wave, text)
                        if reason:
                            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                            continue
                        seen_texts.add(f"{voice}:{key}")
                        text_id = key[:16]
                        duration = len(wave) / TARGET_RATE
                        rows.append({
                            "id": f"{text_id}_{voice}",
                            "text_id": text_id,
                            "voice_id": voice,
                            "text": text,
                            "split": split_of(text),
                            "category": "reading",
                            "origin": repo,
                            "source_id": f"{shard.rfilename}:{stats['seen']}",
                            "source_license": licence,
                            "source_url": f"https://huggingface.co/datasets/{repo}",
                            "duration_seconds": duration,
                            "sample_rate": TARGET_RATE,
                            "asr_transcript": None,
                            "audio": {"bytes": encode(wave), "path": None},
                        })
                        batch_keys.append((voice, duration))
                        kept_voice += 1
                        seconds_voice += duration
                        stats["kept"] += 1
                    if len(rows) >= 256:
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                        rows = []
                    if seconds_voice >= per_voice_cap:
                        break
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows = []
                total_seconds = completed_seconds + seconds_voice
                stats["per_voice"][voice] = {"clips": kept_voice, "hours": round(seconds_voice / 3600, 2)}
                print(f"  {voice}: {kept_voice} clips, {seconds_voice/3600:.2f} h "
                      f"(total {total_seconds/3600:.2f} h)", flush=True)
                if seconds_voice >= per_voice_cap:
                    break
            completed_seconds = total_seconds
    finally:
        writer.close()

    stats["hours"] = round(total_seconds / 3600, 2)
    print(json.dumps({"kept": stats["kept"], "hours": stats["hours"],
                      "rejected": stats["rejected"], "per_voice": stats["per_voice"]},
                     ensure_ascii=False, indent=2))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", nargs="+", default=["MK", "NK"], choices=sorted(DATASETS))
    parser.add_argument("--shards", type=int, default=1, help="parquet shards per voice; 0 = all")
    parser.add_argument("--max-hours", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("data/tr/train.parquet"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stats = run(args.voices, args.shards, args.max_hours, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
