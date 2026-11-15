"""Check that each clip's text matches what is actually said.

The source corpora carry book text, which includes OCR and transcription errors
(``Frankçe``, ``daktili``, ``İskemli'ye``). A TTS model trained on mismatched
text/audio stutters. Policy, per clip:

* WER <= 0.10  -> keep the source text
* 0.10 < WER <= 0.25 -> use the ASR transcript as the label
* WER > 0.25   -> drop the clip

Writes a new parquet with ``text`` replaced where needed, plus ``asr_transcript``,
``wer`` and ``qc`` columns.

    uv run python scripts/verify_transcripts.py --input data/tr/full.parquet \
        --output data/tr/full.checked.parquet
"""

import argparse
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

KEEP = 0.10
REPLACE = 0.25
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_TURKISH_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def normalize(text):
    """Comparison form: Turkish-aware lowercase, no punctuation, single spaces."""
    text = unicodedata.normalize("NFKC", text).translate(_TURKISH_LOWER).lower()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_case(text):
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…":
        text += "."
    return text


def word_error_rate(reference, hypothesis):
    import jiwer

    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0
    return float(jiwer.wer(reference, hypothesis))


def resample_to_16k(wave):
    import soxr

    return soxr.resample(wave, 24000, 16000, quality="HQ")


def load_audio(row):
    import soundfile as sf

    audio = row["audio"]
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    wave, rate = sf.read(source, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != 24000:
        raise ValueError(f"expected 24 kHz parquet, got {rate}")
    return resample_to_16k(wave)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--limit", type=int, help="only the first N clips (for a pilot)")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from faster_whisper import WhisperModel

    print(f"loading faster-whisper {args.model} (int8, cpu) ...", flush=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    reader = pq.ParquetFile(args.input)
    schema = reader.schema_arrow
    fields = list(schema)
    for name, kind in (("asr_transcript", pa.string()), ("wer", pa.float64()), ("qc", pa.string())):
        if name not in schema.names:
            fields.append(pa.field(name, kind))
    out_schema = pa.schema(fields)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"seen": 0, "kept": 0, "replaced": 0, "dropped": 0, "wers": []}
    writer = pq.ParquetWriter(args.output, out_schema)
    try:
        for batch in reader.iter_batches(batch_size=8):
            rows = []
            for row in batch.to_pylist():
                if args.limit and stats["seen"] >= args.limit:
                    break
                stats["seen"] += 1
                try:
                    wave = load_audio(row)
                except Exception:  # noqa: BLE001 - count and continue
                    stats["dropped"] += 1
                    continue
                segments, _ = model.transcribe(wave, language="tr", beam_size=5, vad_filter=False)
                hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
                wer = word_error_rate(row["text"], hypothesis)
                stats["wers"].append(wer)

                if wer <= KEEP:
                    row["qc"] = "keep"
                    stats["kept"] += 1
                elif wer <= REPLACE and normalize(hypothesis):
                    row["text"] = sentence_case(hypothesis)
                    row["qc"] = "asr-text"
                    stats["replaced"] += 1
                else:
                    stats["dropped"] += 1
                    continue
                row["asr_transcript"] = hypothesis
                row["wer"] = wer
                rows.append(row)
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=out_schema))
            if args.limit and stats["seen"] >= args.limit:
                break
    finally:
        writer.close()

    wers = sorted(stats["wers"])
    summary = {
        "seen": stats["seen"],
        "kept": stats["kept"],
        "asr_text": stats["replaced"],
        "dropped": stats["dropped"],
        "drop_rate": round(stats["dropped"] / max(stats["seen"], 1), 3),
        "wer": {
            "median": round(wers[len(wers) // 2], 3) if wers else None,
            "p90": round(wers[int(len(wers) * 0.9)], 3) if wers else None,
            "max": round(wers[-1], 3) if wers else None,
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
