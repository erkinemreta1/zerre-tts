"""Language pack interface.

A language pack owns everything that is language specific: the phoneme
inventory, the eSpeak voice, punctuation, abbreviation and fallback tables, and
the text normalisation steps that run before phonemisation. Everything else in
``zerre`` (acoustics, model, training, export) is language independent.

A process handles one language at a time; the active language is chosen with
``ZERRE_LANG`` (default ``tr``) or by calling :func:`zerre.text.set_language`
before any tensor work starts.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)

STRESS_MARKS = {"ˈ": 1, "ˌ": 2}


def _identity(text: str) -> str:
    return text


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    espeak_voice: str
    consonants: tuple
    vowels: tuple
    punctuation: tuple = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
    # 0 = none, 1 = primary, 2 = secondary. These are the tokens that carry audio.
    voiceless: frozenset = frozenset()
    # eSpeak phonemes outside the inventory map to the closest inventory phones.
    fallback: dict = field(default_factory=dict)
    # Inventory phones the annotation recogniser spells differently (it uses its
    # own eSpeak snapshot, e.g. IPA "ɡ" instead of "g").
    aligner_fallback: dict = field(default_factory=dict)
    abbreviations: frozenset = frozenset()
    # Language specific pre-phonemisation cleanup, applied after the generic one.
    normalize: Callable[[str], str] = _identity
    notes: str = ""

    @property
    def phones(self) -> tuple:
        return tuple(self.consonants) + tuple(self.vowels)

    @property
    def symbols(self) -> tuple:
        return SPECIALS + tuple(self.punctuation) + self.phones

    @property
    def symbol_to_id(self) -> dict:
        return {symbol: index for index, symbol in enumerate(self.symbols)}

    @property
    def phone_set(self) -> frozenset:
        return frozenset(self.phones)

    @property
    def phone_ids(self) -> frozenset:
        index = self.symbol_to_id
        return frozenset(index[phone] for phone in self.phones) | {index[UNK]}

    @property
    def vowel_set(self) -> frozenset:
        return frozenset(self.vowels)

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    def canonical(self, phone: str) -> tuple | None:
        """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
        import unicodedata

        if phone in self.phone_set:
            return (phone,)
        if phone in self.fallback:
            return self.fallback[phone]
        # Some eSpeak voices write a long vowel as a doubled glyph (ɯɯ) instead
        # of the length mark (ɯː); normalise that before giving up.
        if len(phone) == 2 and phone[0] == phone[1]:
            long_form = phone[0] + "ː"
            if long_form in self.phone_set:
                return (long_form,)
        stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
        for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
            if candidate != phone and candidate in self.phone_set:
                return (candidate,)
            if candidate != phone and candidate in self.fallback:
                return self.fallback[candidate]
        return None
