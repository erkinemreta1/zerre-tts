"""Build a word -> phoneme lexicon for the browser demo.

The demo cannot run eSpeak NG (no WASM build is shipped), so the phonemizer is a
lexicon generated here plus a rule fallback for words outside it. Every entry is
produced by the same eSpeak voice and mapped through the same language pack as
training, so known words get exactly the training tokens.

    uv run python scripts/build_lexicon.py --output web/public/lexicon-tr.json
"""

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from zerre.text import SEPARATOR, STRESS_MARKS, canonical, espeak_executable, language

WORD = re.compile(r"^[^\W\d_]+$", re.UNICODE)
BATCH = 64


def texts_from_parquet(path, limit):
    import pyarrow.parquet as pq

    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
        for row in batch.to_pylist():
            yield row["text"]
            seen += 1
            if seen >= limit:
                return


def words_of(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return [word for word in re.findall(r"[^\W\d_]+", text, re.UNICODE) if WORD.match(word)]


def parse(output):
    words = []
    for chunk in output.split():
        units = []
        for unit in chunk.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        words.append(units)
    return words


def phonemize_batch(batch):
    """One eSpeak call for a batch of words; returns a dict word -> (phones, stress)."""
    voice = language().espeak_voice
    process = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", voice, "--stdin"],
        input="\n".join(batch) + "\n", text=True, encoding="utf-8", errors="replace", capture_output=True, check=True)
    parsed = parse(process.stdout)
    if len(parsed) != len(batch):
        return {}
    entries = {}
    for word, units in zip(batch, parsed):
        phones, stress = [], []
        for phone, level in units:
            mapped = canonical(phone)
            if mapped is None:
                continue
            carrier = next((i for i, item in enumerate(mapped) if item in language().vowel_set), 0)
            for index, item in enumerate(mapped):
                phones.append(item)
                stress.append(level if index == carrier else 0)
        if phones:
            entries[word] = (phones, stress)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/tr/full.parquet"))
    parser.add_argument("--limit-texts", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path("web/public/lexicon-tr.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    words = set()
    for text in texts_from_parquet(args.corpus, args.limit_texts):
        words.update(words_of(text))
    ordered = sorted(words)
    print(f"{len(ordered)} distinct words", flush=True)

    lexicon = {}
    for start in range(0, len(ordered), BATCH):
        batch = ordered[start:start + BATCH]
        entries = phonemize_batch(batch)
        if not entries:
            for word in batch:  # one by one when eSpeak regroups the batch
                entries.update(phonemize_batch([word]))
        lexicon.update(entries)
        if start % (BATCH * 40) == 0:
            print(f"  {start + len(batch)}/{len(ordered)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": language().code,
        "voice": language().espeak_voice,
        "symbols": list(language().symbols),
        "words": {word: [" ".join(phones), "".join(str(level) for level in stress)]
                  for word, (phones, stress) in lexicon.items()},
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    report = {"words": len(ordered), "entries": len(lexicon),
              "coverage": round(len(lexicon) / max(len(ordered), 1), 4),
              "bytes": args.output.stat().st_size}
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
