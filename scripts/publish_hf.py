"""Publish the trained model to the Hugging Face Hub.

Needs a token first:

    uv run hf auth login          # or: uv run hf auth login --token hf_xxx

Then:

    uv run python scripts/publish_hf.py --repo erkinemreta1/zerre-tts --dry-run
    uv run python scripts/publish_hf.py --repo erkinemreta1/zerre-tts
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "tr-v1"
SAMPLES = ROOT / "samples"


def collect():
    """Local files mapped to their path inside the repository."""
    card = ROOT / "docs" / "model_card.md"
    files = [
        (card, "README.md"),
        (RUN / "inference.pt", "inference.pt"),
        (RUN / "model.onnx", "onnx/model.onnx"),
        (RUN / "model-int8.onnx", "onnx/model-int8.onnx"),
        (RUN / "model.json", "onnx/model.json"),
        (RUN / "metrics.jsonl", "metrics/metrics.jsonl"),
        (RUN / "gpu.json", "metrics/gpu.json"),
        (RUN / "config.json", "metrics/config.json"),
        (RUN / "recipe.json", "metrics/recipe.json"),
        (ROOT / "LICENSE", "LICENSE"),
        (ROOT / "NOTICE", "NOTICE"),
    ]
    for path in sorted(SAMPLES.glob("taste-step65000-*polished.wav")):
        files.append((path, f"samples/{path.name}"))
    for path in sorted(SAMPLES.glob("taste-step65000-*[!d].wav")):
        if "polished" not in path.name:
            files.append((path, f"samples/{path.name}"))
    return [(source, target) for source, target in files if source.exists()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="erkinemreta1/zerre-tts")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    files = collect()
    missing = [str(path) for path in (RUN / "inference.pt", RUN / "model.onnx", ROOT / "docs" / "model_card.md")
               if not path.exists()]
    if missing:
        raise SystemExit("missing inputs: " + ", ".join(missing))

    total = sum(path.stat().st_size for path, _ in files)
    print(f"{len(files)} dosya, {total / 1e6:.1f} MB → {args.repo}")
    for source, target in files:
        print(f"  {target:44s} {source.stat().st_size / 1e6:8.2f} MB  ({source.relative_to(ROOT)})")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as error:  # noqa: BLE001 - the usual cause is "not logged in"
        raise SystemExit(f"not logged in: {error}\nrun: uv run hf auth login") from None
    print(f"logged in as {who.get('name')}")

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for source, target in files:
        api.upload_file(path_or_fileobj=str(source), path_in_repo=target,
                        repo_id=args.repo, repo_type="model")
        print(f"uploaded {target}")
    print(f"done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
