"""Prepare acoustic features in parallel, then merge the parts.

``zerre.prepare`` spends almost all of its time in ``librosa.pyin``; it is one
process and one core. This driver slices the source parquet into interleaved
parts (so every part carries both voices), runs one ``zerre.prepare`` per part in
a subprocess, and merges the manifests and feature files into one directory that
``zerre.annotate`` accepts unchanged.

    uv run python scripts/prepare_parallel.py --inputs data/tr/full.parquet \
        --output data/tr/features_full --workers 7
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


def write_parts(table, parts_dir, count):
    parts_dir.mkdir(parents=True, exist_ok=True)
    buckets = [[] for _ in range(count)]
    for index in range(table.num_rows):
        buckets[index % count].append(index)
    paths = []
    for number, indices in enumerate(buckets):
        shard = table.take(indices)
        path = parts_dir / f"part{number:02d}.parquet"
        pq.write_table(shard, path, compression="zstd")
        paths.append(path)
        print(f"part {number}: {shard.num_rows} rows -> {path.name}", flush=True)
    return paths


def run_parts(parts, work_dir, split, workers):
    running = []
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(parts)
    while pending or running:
        while pending and len(running) < workers:
            part = pending.pop(0)
            name = part.stem
            output = work_dir / name
            log = (log_dir / f"{name}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "zerre.prepare", "--inputs", str(part),
                 "--split", split, "--output", str(output)],
                stdout=log, stderr=subprocess.STDOUT, cwd=Path.cwd(),
            )
            running.append((name, process, log))
            print(f"started {name} (pid {process.pid})", flush=True)
        time.sleep(5)
        still = []
        for name, process, log in running:
            code = process.poll()
            if code is None:
                still.append((name, process, log))
                continue
            log.close()
            print(f"finished {name} with exit code {code}", flush=True)
            if code != 0:
                raise SystemExit(f"part {name} failed; see {work_dir / 'logs' / (name + '.log')}")
        running = still


def merge(work_dir, output, split, parts):
    train_dir = output / split
    train_dir.mkdir(parents=True, exist_ok=True)
    merged, rejected, accepted, dropped = [], [], 0, 0
    for part in parts:
        name = part.stem
        part_dir = work_dir / name
        manifest = part_dir / f"{split}.jsonl"
        if not manifest.exists():
            raise SystemExit(f"{manifest} is missing; part did not finish")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = part_dir / row["path"]
            target = output / row["path"]
            if not target.exists():
                shutil.move(str(source), str(target))
            merged.append(row)
        rejects = part_dir / f"{split}.rejected.jsonl"
        if rejects.exists():
            rejected.extend(line for line in rejects.read_text(encoding="utf-8").splitlines() if line.strip())
        dropped += len(rejected)
        shutil.rmtree(part_dir / split, ignore_errors=True)
        manifest.unlink(missing_ok=True)
        rejects.unlink(missing_ok=True)
        (part_dir / f"{split}.preparation.json").unlink(missing_ok=True)

    with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / f"{split}.rejected.jsonl").open("w", encoding="utf-8") as stream:
        for line in rejected:
            stream.write(line + "\n")

    features_json = None
    for part in parts:
        candidate = work_dir / part.stem / "features.json"
        if candidate.exists():
            features_json = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if features_json is not None:
        (output / "features.json").write_text(json.dumps(features_json), encoding="utf-8")
    (output / f"{split}.preparation.json").write_text(json.dumps({
        "identity": {"merged_parts": [part.stem for part in parts], "features": features_json},
        "complete": True,
        "accepted": accepted,
        "rejected": dropped,
    }), encoding="utf-8")
    return accepted, dropped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args(argv)

    table = pq.read_table(args.inputs)
    work_dir = args.output.parent / (args.output.name + ".parts")
    work_dir.mkdir(parents=True, exist_ok=True)
    parts = write_parts(table, work_dir, args.workers)
    run_parts(parts, work_dir, args.split, args.workers)
    accepted, rejected = merge(work_dir, args.output, args.split, parts)
    print(json.dumps({"accepted": accepted, "rejected": rejected, "output": str(args.output)}, indent=2))
    if not args.keep_parts:
        for part in parts:
            part.unlink(missing_ok=True)
        (work_dir / "logs").exists() and shutil.rmtree(work_dir / "logs", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
