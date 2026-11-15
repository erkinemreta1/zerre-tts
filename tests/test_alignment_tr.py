import json
from pathlib import Path

import pytest

from zerre.align import ALIGNER_MODEL, label_fallback
from zerre.text import PHONE_SET, UNK, language


def test_fallback_has_unknown_and_turkish_velar_stop():
    fallback = label_fallback()
    assert fallback[UNK] == "<unk>"
    assert language().code == "tr"
    # the recogniser's eSpeak snapshot writes IPA ɡ (U+0261), eSpeak NG 1.52 writes g
    assert fallback["g"] == "ɡ"


def test_every_inventory_phone_has_a_recogniser_label():
    """Needs the aligner vocabulary in the local HF cache; skipped otherwise."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(ALIGNER_MODEL, "vocab.json", local_files_only=True)
    except Exception:  # noqa: BLE001 - cache miss is not a test failure
        pytest.skip("aligner vocabulary is not in the HF cache")
    vocabulary = json.loads(Path(path).read_text(encoding="utf-8"))
    fallback = label_fallback()
    missing = sorted(phone for phone in PHONE_SET if phone not in vocabulary and phone not in fallback)
    assert not missing, f"recogniser cannot label these phones: {missing}"
