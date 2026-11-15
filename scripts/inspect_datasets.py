"""Inspect the Turkish source datasets before committing to a full download.

Reads the smallest parquet shard of each Hugging Face dataset directly instead of
going through ``datasets`` audio decoding (which needs torchcodec). Reports
sample rate, duration distribution, schema and text sanity, and writes a JSON
report.

    uv run python scripts/inspect_datasets.py --rows 100
"""

import argparse
import collections
import io
import json
import statistics
import sys
from pathlib import Path

DATASETS = {
    "MK": "omersaidd/tts_mazlum_kiper_tur",
    "NK": "omersaidd/tts_nisan_kumru_tur",
}


def smallest_shard(repo):
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, files_metadata=True)
    shards = [
        sibling for sibling in info.siblings
        if sibling.rfilename.endswith(".parquet")
    ]
    if not shards:
        raise RuntimeError(f"{repo}: no parquet shards found")
    return min(shards, key=lambda sibling: sibling.size or 0)


def decode(audio):
    import soundfile as sf

    if isinstance(audio, dict) and audio.get("bytes") is not None:
        source = io.BytesIO(audio["bytes"])
    elif isinstance(audio, dict) and audio.get("path"):
        source = audio["path"]
    else:
        raise ValueError("audio column is neither bytes nor a path")
    return sf.read(source, dtype="float32")


def inspect(repo, rows):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shard = smallest_shard(repo)
    local = hf_hub_download(repo, shard.rfilename, repo_type="dataset")
    parquet = pq.ParquetFile(local)
    sample_rates = collections.Counter()
    durations = []
    texts = []
    failures = collections.Counter()
    scanned = 0
    for batch in parquet.iter_batches(batch_size=8, columns=None):
        for row in batch.to_pylist():
            if scanned >= rows:
                break
            scanned += 1
            try:
                wave, rate = decode(row["audio"])
            except Exception as error:  # noqa: BLE001 - report, keep going
                failures[f"{type(error).__name__}: {error}"] += 1
                continue
            sample_rates[rate] += 1
            durations.append(len(wave) / rate)
            texts.append((row.get("text") or "").strip())
        if scanned >= rows:
            break
    return {
        "repo": repo,
        "shard": shard.rfilename,
        "shard_bytes": shard.size,
        "columns": parquet.schema_arrow.names,
        "rows_scanned": scanned,
        "sample_rates": dict(sample_rates),
        "decode_failures": dict(failures),
        "duration_s": {
            "min": round(min(durations), 2) if durations else None,
            "median": round(statistics.median(durations), 2) if durations else None,
            "max": round(max(durations), 2) if durations else None,
            "total": round(sum(durations), 1),
        },
        "empty_texts": sum(1 for text in texts if not text),
        "text_chars": {
            "min": min((len(text) for text in texts), default=0),
            "median": int(statistics.median([len(text) for text in texts])) if texts else 0,
            "max": max((len(text) for text in texts), default=0),
        },
        "examples": texts[:3],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = {}
    for voice, repo in DATASETS.items():
        print(f"inspecting {voice}: {repo} ...", flush=True)
        report[voice] = inspect(repo, args.rows)
        print(json.dumps(report[voice], ensure_ascii=False, indent=2), flush=True)

    problems = [
        f"{voice}: sample rates {info['sample_rates']} (expected only 24000)"
        for voice, info in report.items()
        if set(info["sample_rates"]) != {24000}
    ]
    for line in problems:
        print(line)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
