"""Apply the transcript QC result to an existing feature manifest.

Re-running ``zerre.prepare`` after transcript QC would recompute hours of DSP for
nothing. This script instead rewrites the manifest in place:

* ``qc == "keep"``       -> the row stays untouched
* ``qc == "asr-text"``   -> ``text`` and ``text_hash`` are replaced
* row missing / dropped -> the manifest row and its feature file are removed

    uv run python scripts/apply_qc.py --features data/tr/features_full \
        --checked data/tr/full.checked.parquet --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

from zerre.qc import plan, text_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="directory holding <split>.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--checked", type=Path, required=True, help="parquet from verify_transcripts.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq

    manifest_path = args.features / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qc_rows = pq.read_table(args.checked, columns=["id", "text", "qc"]).to_pylist()

    updates, dropped = plan(rows, qc_rows)
    kept = []
    removed_files = 0
    for row in rows:
        if row["id"] in dropped:
            feature = args.features / row["path"]
            if feature.is_file() and not args.dry_run:
                feature.unlink()
                removed_files += 1
            continue
        if row["id"] in updates:
            row["text"] = updates[row["id"]]
            row["text_hash"] = text_hash(row["text"])
        kept.append(row)

    summary = {
        "manifest": str(manifest_path),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "text_replaced": len(updates),
        "dropped": len(dropped),
        "files_removed": removed_files,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        temporary = manifest_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        rejected = args.features / f"{args.split}.qc-dropped.jsonl"
        with rejected.open("w", encoding="utf-8") as stream:
            for identifier in dropped:
                stream.write(json.dumps({"id": identifier}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
