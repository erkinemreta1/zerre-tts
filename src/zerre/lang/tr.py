"""Turkish language pack.

Inventory follows eSpeak NG's ``tr`` voice as printed by
``espeak-ng -q --ipa --sep=_ -v tr``. The phones are listed in a fixed order and
must stay append-only once a checkpoint exists: inserting a phone shifts every
token id after it.

``scripts/audit_inventory.py`` re-derives the inventory from a Turkish corpus and
reports any eSpeak phone this table does not know. Run it after changing the
eSpeak version pinned in ``pyproject.toml``/``Dockerfile``.
"""

import re

from .base import Language

# Turkish consonants as eSpeak renders them. Palatal /c, ɟ/ and velar /k, g/
# contrast before front and back vowels, so both are separate phones.
CONSONANTS = (
    "p", "b", "t", "d", "c", "ɟ", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ɲ", "ŋ", "l", "ɫ", "ɾ", "r", "j", "w", "tʃ", "dʒ",
)

# Eight short vowels plus the long variants that appear with â, î, û and with
# the silent ğ between vowels (dağ -> daː, öğretmen -> œː). eSpeak renders ü as
# both /y/ and /ø/ depending on the word, and ö as /œ/; they stay separate.
# [ɛ] and [æ] are the contextual realisations of e, mapped onto /e/ by FALLBACK.
VOWELS = (
    "a", "e", "i", "o", "u", "y", "ø", "œ", "ɯ",
    "aː", "eː", "iː", "oː", "uː", "yː", "øː", "œː", "ɯː",
)

# eSpeak phones from loanwords and foreign names pushed onto the closest Turkish
# phone. Anything still unknown becomes <unk> and is reported by the audit.
FALLBACK = {
    "r": ("ɾ",), "ɹ": ("ɾ",), "ʀ": ("ɾ",),
    # eSpeak NG 1.52 spells the velar stop both ways depending on the word.
    "ɡ": ("g",),
    # Long ı comes out as a doubled glyph rather than the length mark.
    "ɯɯ": ("ɯː",),
    "θ": ("t",), "ð": ("d",), "ɣ": ("g",), "x": ("h",), "q": ("k",),
    "ʎ": ("l", "j"), "ɬ": ("l",), "ts": ("t", "s"), "dz": ("d", "z"),
    "β": ("v",), "ʋ": ("v",), "ɸ": ("f",), "ʂ": ("ʃ",), "ʐ": ("ʒ",),
    "ə": ("a",), "ɛ": ("e",), "æ": ("e",), "ɪ": ("i",), "ɔ": ("o",), "ʊ": ("u",),
    "ʌ": ("a",), "ɐ": ("a",), "ɜ": ("e",), "ɨ": ("ɯ",),
    "ʏ": ("y",), "ɑ": ("a",), "ɒ": ("a",), "ɵ": ("œ",),
    "pʰ": ("p",), "tʰ": ("t",), "kʰ": ("k",), "cʰ": ("c",), "tʃʰ": ("tʃ",),
}

ABBREVIATIONS = frozenset({
    "dr", "prof", "doç", "av", "vb", "vs", "bkz", "örn", "mah", "cad", "sok",
    "no", "tel", "tk", "tbmm", "trt", "ab", "abd", "bk", "yy", "yy.",
})


def normalize(text: str) -> str:
    """Turkish specifics applied after the generic cleanup in ``zerre.text``.

    * Proper-noun suffixes take an apostrophe in writing but are read as one
      word: ``Ankara'da`` -> ``Ankarada``.
    * A suffix after a digit keeps a space, otherwise eSpeak sees a non-number
      token: ``1923'te`` -> ``1923 te``.
    * ``&``, ``%`` and the lira sign are spoken forms.
    """
    text = re.sub(r"(?<=[^\W\d_])['’](?=[^\W\d_])", "", text)
    text = re.sub(r"(?<=\d)['’](?=[^\W\d_])", " ", text)
    text = re.sub(r"\s*&\s*", " ve ", text)
    text = re.sub(r"%\s*(?=\d)", "yüzde ", text)
    text = text.replace("₺", " TL ")
    return re.sub(r"\s+", " ", text).strip()


LANGUAGE = Language(
    code="tr",
    name="Turkish",
    espeak_voice="tr",
    consonants=CONSONANTS,
    vowels=VOWELS,
    voiceless=frozenset(("p", "t", "c", "k", "f", "s", "ʃ", "h", "tʃ")),
    fallback=FALLBACK,
    abbreviations=ABBREVIATIONS,
    normalize=normalize,
    # The annotation recogniser's eSpeak snapshot spells the velar stop with the
    # IPA letter U+0261, while this inventory and eSpeak NG 1.52 use ASCII "g".
    aligner_fallback={"g": "ɡ", "ɯː": "ɯ"},
    notes="Stress is final-syllable by default; eSpeak marks the exceptions.",
)
