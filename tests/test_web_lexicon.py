import json
from pathlib import Path

import pytest

from zerre import lang
from zerre.text import espeak_executable, phonemize

LEXICON = Path(__file__).resolve().parents[1] / "web" / "public" / "lexicon-tr.json"


@pytest.fixture(scope="module")
def lexicon():
    if not LEXICON.exists():
        pytest.skip("web lexicon has not been built")
    return json.loads(LEXICON.read_text(encoding="utf-8"))


def test_lexicon_uses_the_training_inventory(lexicon):
    assert lexicon["language"] == lang.current_code()
    assert lexicon["symbols"] == list(lang.current().symbols)


def test_every_lexicon_phone_is_a_symbol(lexicon):
    symbols = set(lexicon["symbols"])
    unknowns = set()
    for phones, _ in lexicon["words"].values():
        unknowns.update(phone for phone in phones.split(" ") if phone and phone not in symbols)
    assert not unknowns, f"lexicon has phones outside the inventory: {sorted(unknowns)}"


def test_stress_string_length_matches_phone_count(lexicon):
    for word, (phones, stress) in lexicon["words"].items():
        assert len(phones.split(" ")) == len(stress), word


def test_known_word_matches_espeak(lexicon):
    entry = lexicon["words"].get("merhaba")
    if entry is None:
        pytest.skip("word not in the lexicon")
    try:
        espeak_executable()
    except RuntimeError:
        pytest.skip("eSpeak NG is not installed")
    phones = phonemize("merhaba")
    stored = [phone for phone in entry[0].split(" ") if phone]
    expected = [symbol for symbol in phones.symbols if symbol in lang.current().phone_set]
    assert stored == expected
