"""Text front end: text -> phoneme tokens with a stress feature.

The same code runs during annotation, dataset preparation, training and
inference, so tokens can never drift between them. Phonemes come from eSpeak NG
(``--ipa --sep=_``), which is the only pronunciation source: an external
rule-based program, no neural G2P, no downloaded weights.

* A token is one eSpeak phoneme exactly as eSpeak segments it, for example
  ``t``, ``a``, ``ʃ`` or ``y``. This is also the label inventory of the
  phoneme recogniser used for offline forced alignment, so one aligned label
  corresponds to exactly one token.
* Lexical stress is a parallel feature (0 none, 1 primary, 2 secondary), not a
  token, so it never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.

Everything language specific — inventory, eSpeak voice, punctuation,
abbreviations, fallback table, normalisation — lives in ``zerre.lang``. This
module only knows how to drive eSpeak and how to turn its output into ids.

eSpeak NG 1.51 drops the last character of unterminated standard input, so the
input always ends with a newline (regression test in ``tests/test_frontend.py``).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata

from . import lang
from .lang.base import (
    BOS,
    EOS,
    PAD,
    SP,
    SPECIALS,
    STRESS_MARKS,
    UNK,
    Language,
)

__all__ = [
    "BOS", "EOS", "PAD", "PHONES", "PHONE_IDS", "PHONE_SET", "PUNCTUATION", "SP",
    "SPECIALS", "SYMBOLS", "SYMBOL_TO_ID", "UNK", "VOICELESS", "VOWELS",
    "NothingToPronounce", "Phonemized", "frontend", "frontend_fingerprint",
    "frontend_identity", "frontend_runtime", "is_phone", "language",
    "normalize_text", "phonemize", "set_frontend", "set_language", "tokenize",
]

FRONTEND_VERSION = 5  # 4: English + dizione; 5: language packs, eSpeak only.
SEPARATOR = "_"
FRONTENDS = ("espeak",)
FRONTEND_VARIABLE = "ZERRE_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "espeak")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

_LANGUAGE: Language
SYMBOLS: tuple
SYMBOL_TO_ID: dict
PHONES: tuple
PHONE_SET: frozenset
PHONE_IDS: frozenset
VOWELS: tuple
VOICELESS: frozenset
VOWEL_SET: frozenset
PUNCTUATION: tuple
_PUNCTUATION_CHARS: frozenset

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")
_WORD_END = re.compile(r"([^\W\d_]+)$")


class NothingToPronounce(ValueError):
    """The text has no word, number or letter to say: only marks, emoji and the like."""


class Phonemized:
    __slots__ = ("ids", "stress", "symbols", "unknown")

    def __init__(self, symbols, ids, stress, unknown):
        self.symbols = symbols
        self.ids = ids
        self.stress = stress
        self.unknown = unknown

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"Phonemized({len(self.ids)} tokens)"


def _bind(language: Language) -> None:
    """Point the module-level token tables at ``language``."""
    global _LANGUAGE, SYMBOLS, SYMBOL_TO_ID, PHONES, PHONE_SET, PHONE_IDS
    global VOWELS, VOICELESS, VOWEL_SET, PUNCTUATION, _PUNCTUATION_CHARS
    _LANGUAGE = language
    SYMBOLS = language.symbols
    SYMBOL_TO_ID = language.symbol_to_id
    PHONES = language.phones
    PHONE_SET = language.phone_set
    PHONE_IDS = language.phone_ids
    VOWELS = language.vowels
    VOICELESS = language.voiceless
    VOWEL_SET = language.vowel_set
    PUNCTUATION = language.punctuation
    _PUNCTUATION_CHARS = frozenset(PUNCTUATION)


def language() -> Language:
    return _LANGUAGE


def set_language(code: str) -> Language:
    return lang.set_language(code)


_bind(lang.current())


def frontend():
    """Name of the program that produces the phonemes."""
    return _frontend


def set_frontend(name):
    """Select the phonemizer for this process. Tokens of different front ends
    must not be mixed: annotations and checkpoints record which one made them."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def frontend_identity():
    return {
        "frontend_version": FRONTEND_VERSION,
        "language": _LANGUAGE.code,
        "voice": _LANGUAGE.espeak_voice,
        "separator": SEPARATOR,
        "symbols": list(SYMBOLS),
        "fallback": {key: list(value) for key, value in sorted(_LANGUAGE.fallback.items())},
    }


def frontend_runtime():
    """What an annotation run records about the exact program behind its tokens."""
    return {"espeak_version": espeak_version()}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _LANGUAGE.normalize(text)


def chunks(text):
    """Yield ``(is_punctuation, piece)``. Decimals, clock times, digit groups,
    titles and initialisms stay inside the text piece for eSpeak; every other
    punctuation mark becomes its own token."""
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i - 1] if i else ""
        following = text[i + 1] if i + 1 < len(text) else ""
        if char == "'" and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == ".":
            word = _WORD_END.search(text[:i])
            if word and word[1].lower() in _LANGUAGE.abbreviations:
                continue
            if previous.isalpha() and following.isalpha():
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i + 1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    """Locate eSpeak NG: ``ZERRE_ESPEAK``, then PATH, then the usual install dirs."""
    override = os.environ.get("ZERRE_ESPEAK")
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"ZERRE_ESPEAK points at {override!r}, which is not a file")
        return override
    for name in ("espeak-ng", "espeak-ng.exe"):
        found = shutil.which(name)
        if found:
            return found
    candidates = (
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
        "/opt/homebrew/bin/espeak-ng",
        "/usr/local/bin/espeak-ng",
        "/usr/bin/espeak-ng",
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        "eSpeak NG not found. Install it (brew install espeak-ng, apt install espeak-ng, "
        "winget install eSpeak-NG.eSpeak-NG) or set ZERRE_ESPEAK to the executable."
    )


def espeak_version():
    result = subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=True, timeout=10)
    return result.stdout.strip()


def espeak_words(piece):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``.

    The input must end with a newline: eSpeak NG 1.51 silently drops the last
    character of unterminated standard input.
    """
    result = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", _LANGUAGE.espeak_voice, "--stdin"],
        input=piece.rstrip("\n") + "\n", text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=True, timeout=30)
    return parse_espeak(result.stdout)


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone):
    """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
    return _LANGUAGE.canonical(phone)


def tokenize(pieces):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is the parsed eSpeak word list. Pure function (testable
    without eSpeak)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                # A fallback may expand to several phones: stress goes to its first vowel.
                carrier = next((i for i, item in enumerate(mapped) if item in VOWEL_SET), 0)
                for index, item in enumerate(mapped):
                    symbols.append(item)
                    stress.append(level if index == carrier else 0)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    pieces = []
    for punctuation, piece in chunks(normalize_text(text)):
        pieces.append((True, piece) if punctuation else (False, espeak_words(piece)))
    return tokenize(pieces)


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
