import pytest

from zerre import lang
from zerre.text import (
    BOS,
    EOS,
    SP,
    UNK,
    espeak_executable,
    normalize_text,
    phonemize,
    tokenize,
)

TR = lang.get("tr")


def espeak_available():
    try:
        espeak_executable()
    except RuntimeError:
        return False
    return True


needs_espeak = pytest.mark.skipif(not espeak_available(), reason="eSpeak NG is not installed")


# --- normalisation ---------------------------------------------------------

def test_proper_noun_suffix_loses_apostrophe():
    assert TR.normalize("Ankara'da yaşıyorum") == "Ankarada yaşıyorum"


def test_apostrophe_after_digit_becomes_space():
    assert TR.normalize("1923'te kuruldu") == "1923 te kuruldu"


def test_percent_and_ampersand_are_spoken():
    assert TR.normalize("indirim %25") == "indirim yüzde 25"
    assert TR.normalize("Ali & Ayşe") == "Ali ve Ayşe"


def test_generic_normalisation_runs_first():
    text = normalize_text("  Çok   güzel...  ")
    assert text == "Çok güzel…"


# --- tokenisation (pure, no eSpeak) ---------------------------------------

def test_tokenize_builds_boundaries_word_gaps_and_stress():
    pieces = [
        (False, [[("m", 0), ("e", 1), ("r", 0)]]),
        (False, [[("h", 0), ("a", 0), ("b", 0), ("a", 0)]]),
        (True, "!"),
    ]
    phones = tokenize(pieces)
    assert phones.symbols[0] == BOS and phones.symbols[-1] == EOS
    assert SP in phones.symbols
    assert "!" in phones.symbols
    # stress sits on the first vowel of the word that eSpeak marked
    stressed = [symbol for symbol, level in zip(phones.symbols, phones.stress) if level]
    assert stressed == ["e"]


def test_tokenize_keeps_ids_and_symbols_in_step():
    phones = tokenize([(False, [[("a", 0)]])])
    from zerre.text import SYMBOL_TO_ID

    assert all(SYMBOL_TO_ID[symbol] == index for symbol, index in zip(phones.symbols, phones.ids))


def test_repeated_punctuation_collapses():
    phones = tokenize([(False, [[("a", 0)]]), (True, "!"), (True, "!")])
    assert phones.symbols.count("!") == 1


def test_unknown_phone_is_reported_not_dropped():
    phones = tokenize([(False, [[("ʘ", 0)]])])
    assert UNK in phones.symbols
    assert phones.unknown == ("ʘ",)


def test_nothing_to_pronounce():
    from zerre.text import NothingToPronounce

    with pytest.raises(NothingToPronounce):
        tokenize([(True, "!")])


# --- inventory ------------------------------------------------------------

def test_inventory_is_append_only_and_unique():
    assert len(set(TR.symbols)) == len(TR.symbols)
    assert TR.symbols[:5] == ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>")
    assert TR.vocab_size == len(TR.symbols)


def test_canonical_maps_foreign_phone_to_inventory():
    assert TR.canonical("ɹ") == ("ɾ",)
    assert TR.canonical("θ") == ("t",)
    assert TR.canonical("ʘ") is None
    assert TR.canonical("a") == ("a",)


# --- integration with eSpeak ---------------------------------------------

@needs_espeak
def test_phonemize_produces_turkish_phonemes():
    phones = phonemize("Merhaba dünya!")
    assert phones.symbols[0] == BOS and phones.symbols[-1] == EOS
    assert "!" in phones.symbols
    assert not phones.unknown
    # ü and ö must stay front rounded vowels with their own phones
    assert "y" in phones.symbols or "ø" in phones.symbols
    assert phonemize("öğretmen").symbols[1] in {"œ", "œː"}


@needs_espeak
def test_phonemize_survives_numbers_and_abbreviations():
    phones = phonemize("Prof. Dr. Ahmet, saat 14:30'da 1.500 TL ödedi.")
    assert not phones.unknown
    assert len(phones) > 10


@needs_espeak
def test_phonemize_rejects_empty_text():
    with pytest.raises(ValueError):
        phonemize("   ")
