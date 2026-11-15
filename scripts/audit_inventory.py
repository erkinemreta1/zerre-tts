"""Verify the active language pack's phoneme inventory against eSpeak NG.

eSpeak changes its output between releases, and a phone the pack does not know
silently becomes ``<unk>`` in training data. Run this after upgrading eSpeak or
after editing a language pack.

    uv run python scripts/audit_inventory.py                # built-in sample
    uv run python scripts/audit_inventory.py corpus.txt     # your own text

Exit code 1 when a phone has no inventory entry and no fallback mapping.
"""

import argparse
import collections
import sys
from pathlib import Path

from zerre.text import SYMBOLS, language, phonemize

SAMPLE = """
Merhaba! Bu bir Türkçe konuşma sentezi denemesidir.
Ankara'da 1923'te kurulan TBMM, 23 Nisan 1920'de açıldı.
Şu çocuk ğ ile ı arasındaki farkı biliyor mu?
Öğretmen, öğrencilerine güzel bir hikâye okudu; herkes çok sevindi.
Küçük köpek, büyük kediden korktuğu için kaçtı.
Yüzde 25 indirimle 1.500 TL'ye aldım, çok ucuzdu!
Zeynep'in düğününde İzmir'den gelen misafirler vardı.
Prof. Dr. Ahmet Yılmaz, cuma günü saat 14:30'da konuşacak.
Çiçekler çok güzel kokuyordu, arılar etraflarında uçuşuyordu.
Gökyüzünde süzülen kuşlar, akşam serinliğinde yuvalarına döndü.
"""


def sentences(text):
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", type=Path, help="UTF-8 text files; default: built-in sample")
    args = parser.parse_args(argv)

    pack = language()
    if args.texts:
        text = "\n".join(path.read_text(encoding="utf-8") for path in args.texts)
    else:
        text = SAMPLE

    used = collections.Counter()
    unknown = collections.Counter()
    for sentence in sentences(text):
        phones = phonemize(sentence)
        used.update(phones.symbols)
        unknown.update(phones.unknown)

    inventory = set(pack.phone_set)
    seen_phones = {symbol for symbol in used if symbol in inventory}
    unused = sorted(inventory - seen_phones)
    invisible = sorted(s for s in used if s not in set(SYMBOLS))

    print(f"language: {pack.code} ({pack.name}), eSpeak voice {pack.espeak_voice!r}")
    print(f"vocab_size: {pack.vocab_size} ({len(pack.phones)} phones)")
    print(f"phones used in this sample: {len(seen_phones)}")
    if unused:
        print(f"phones in the inventory but unused here: {' '.join(unused)}")
    if unknown:
        print(f"UNMAPPED eSpeak phones: {' '.join(sorted(unknown))}")
        print("Add them to the language pack inventory or its FALLBACK table.")
    if invisible:
        print(f"tokens missing from SYMBOLS: {invisible}")
    if not unknown and not invisible:
        print("inventory OK: every eSpeak phone maps to a token")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
