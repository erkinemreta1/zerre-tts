"""Transcript QC bookkeeping: which feature rows change text and which go away."""

import hashlib


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def plan(manifest_rows, qc_rows):
    """Return ``(updates, dropped)`` from the manifest and the QC parquet rows.

    ``updates`` maps clip id -> replacement text, only for rows whose label the
    ASR pass decided to replace. ``dropped`` lists ids absent from the QC output
    (the clip failed the WER gate or could not be decoded).
    """
    seen = {row["id"]: row for row in qc_rows}
    updates = {}
    dropped = []
    for row in manifest_rows:
        replacement = seen.get(row["id"])
        if replacement is None:
            dropped.append(row["id"])
            continue
        if replacement.get("qc") == "asr-text":
            text = (replacement.get("text") or "").strip()
            if text and text != row["text"]:
                updates[row["id"]] = text
    return updates, dropped
