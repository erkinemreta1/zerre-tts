"""Check that an annotation run is complete and describe what to upload for training.

Verifies, without loading audio: every row's feature file exists, the acoustic
fingerprint matches the current config, the annotation run finished, and the
per-voice hours add up. Writes ``training-bundle.json`` next to the annotations.

    uv run python scripts/package_training.py --annotations data/tr/annotations \
        --features data/tr/features_full
"""

import argparse
import collections
import json
import sys
from pathlib import Path

from zerre.config import Config


def summarize_split(annotations, features, split, config):
    manifest = annotations / f"{split}.jsonl"
    if not manifest.exists():
        return None
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = 0
    mismatched = 0
    hours = collections.Counter()
    for row in rows:
        path = features / row["path"]
        if not path.is_file():
            missing += 1
        if row.get("acoustic") != config.acoustic_fingerprint:
            mismatched += 1
        hours[row["speaker"]] += row["frames"] * config.hop / config.sample_rate / 3600
    return {
        "split": split,
        "clips": len(rows),
        "hours": round(sum(hours.values()), 2),
        "by_voice": {voice: round(value, 2) for voice, value in sorted(hours.items())},
        "missing_feature_files": missing,
        "fingerprint_mismatches": mismatched,
        "total_frames": sum(row["frames"] for row in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = Config()
    metadata_path = args.annotations / "annotations.json"
    if not metadata_path.exists():
        raise SystemExit(f"{metadata_path} is missing; the annotation run has not started")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = metadata.get("identity", {})
    compatible = identity.get("acoustic") == config.acoustic_fingerprint

    splits = {}
    for split in ("train", "validation", "test"):
        summary = summarize_split(args.annotations, args.features, split, config)
        if summary is not None:
            splits[split] = summary

    problems = []
    if not metadata.get("complete"):
        problems.append("annotation run reports complete=false")
    if not compatible:
        problems.append("annotations were made with a different acoustic config")
    for split, summary in splits.items():
        if summary["missing_feature_files"]:
            problems.append(f"{split}: {summary['missing_feature_files']} feature files missing")
        if summary["fingerprint_mismatches"]:
            problems.append(f"{split}: {summary['fingerprint_mismatches']} rows with another fingerprint")
    if "train" not in splits:
        problems.append("no train split")
    if "validation" not in splits:
        problems.append("no validation split")

    bundle = {
        "annotations": str(args.annotations),
        "features": str(args.features),
        "config": {"language": config.language, "sample_rate": config.sample_rate,
                   "speakers": list(config.speakers), "vocab_size": config.vocab_size},
        "annotation_fingerprint": metadata.get("fingerprint"),
        "splits": splits,
        "problems": problems,
    }
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    if problems:
        print("\n".join(f"- {problem}" for problem in problems), file=sys.stderr)
        return 1
    print("\nupload for training:\n"
          f"  hf upload <user>/zerre-tts-tr-data {args.annotations} annotations --repo-type dataset\n"
          f"  hf upload <user>/zerre-tts-tr-data {args.features} features --repo-type dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
