"""Compare the browser phonemizer with eSpeak on real sentences.

The demo cannot run eSpeak, so it uses a corpus lexicon plus rules. This script
measures how close that is to the phonemes the model was trained on.

    uv run python scripts/check_web_phonemes.py --sentences 60
"""

import argparse
import difflib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from zerre.text import phonemize

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / "web" / "public" / "lexicon-tr.json"


def sample(parquet, count):
    import pyarrow.parquet as pq

    sentences = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32, columns=["text"]):
        for row in batch.to_pylist():
            text = row["text"].strip()
            if 40 <= len(text) <= 160 and not any(character.isdigit() for character in text):
                sentences.append(text)
            if len(sentences) >= count:
                return sentences
    return sentences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "tr" / "full.parquet")
    parser.add_argument("--sentences", type=int, default=60)
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    args = parser.parse_args(argv)

    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))
    sentences = sample(args.corpus, args.sentences)
    payload = {"sentences": sentences, "lexicon": str(args.lexicon), "symbols": lexicon["symbols"]}
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.json"
        source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(["node", str(ROOT / "scripts" / "web_phonemes.js"), str(source)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", check=True)
    web = json.loads(result.stdout)

    exact = 0
    ratios = []
    examples = []
    for sentence, produced in zip(sentences, web):
        reference = phonemize(sentence).symbols
        candidate = produced["symbols"]
        if list(reference) == list(candidate):
            exact += 1
        ratio = difflib.SequenceMatcher(a=list(reference), b=list(candidate)).ratio()
        ratios.append(ratio)
        if ratio < 0.9 and len(examples) < 3:
            examples.append({"sentence": sentence[:70], "reference": " ".join(reference)[:110],
                             "web": " ".join(candidate)[:110], "ratio": round(ratio, 3)})

    summary = {
        "sentences": len(sentences),
        "exact_match": exact,
        "exact_rate": round(exact / max(len(sentences), 1), 3),
        "median_token_ratio": round(sorted(ratios)[len(ratios) // 2], 4) if ratios else None,
        "mean_token_ratio": round(sum(ratios) / max(len(ratios), 1), 4),
        "worst": examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["mean_token_ratio"] and summary["mean_token_ratio"] > 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
