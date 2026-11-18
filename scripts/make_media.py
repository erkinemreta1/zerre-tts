"""Render the zerre-tts media assets as SVG.

    uv run python scripts/make_media.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"

INK = "#12141a"
PAPER = "#fbfbfa"
ACCENT = "#0b6b5b"
ACCENT2 = "#5ad2b8"

# One idea per asset: the wordmark IS the waveform (the hyphen becomes a pause,
# "tts" is spoken by the second wave). No letters-as-signs.
LOGO = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 160" width="720" height="160" role="img" aria-label="zerre-tts">
  <rect width="720" height="160" fill="{PAPER}"/>
  <text x="30" y="95" font-family="ui-monospace, 'Cascadia Code', Menlo, monospace" font-size="64" font-weight="700" fill="{INK}">zerre</text>
  <g fill="none" stroke="{ACCENT}" stroke-width="12" stroke-linecap="round">
    <path d="M330 70 l40 0" stroke="{PAPER}"/>
    <path d="M380 45 a30 30 0 1 0 0.1 0"/>
    <path d="M470 55 v50 m0 -25 q18 -18 30 6 l10 19"/>
    <path d="M550 55 v50 m0 -25 q18 -18 30 6 l10 19"/>
    <path d="M630 55 v50 m0 -25 q18 -18 30 6 l10 19"/>
  </g>
  <text x="30" y="140" font-family="ui-monospace, Menlo, monospace" font-size="22" fill="{ACCENT}">8.5 MB · tarayıcıda · Türkçe TTS</text>
</svg>
"""

OG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630" role="img" aria-label="zerre-tts">
  <rect width="1200" height="630" fill="{PAPER}"/>
  <rect x="0" y="520" width="1200" height="110" fill="{ACCENT}" opacity="0.10"/>
  <text x="90" y="200" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="96" font-weight="700" fill="{INK}">zerre-tts</text>
  <text x="90" y="280" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="44" fill="{ACCENT}">8.5 MB Turkish TTS, trained from scratch</text>
  <g fill="none" stroke="{ACCENT}" stroke-width="11" stroke-linecap="round">
    <path d="M90 430 q45 -140 90 0 q45 -140 90 0"/>
    <path d="M330 430 v-120 m0 60 q40 -40 62 10 l20 50"/>
    <path d="M480 430 v-120 m0 60 q40 -40 62 10 l20 50"/>
    <path d="M630 430 v-120 m0 60 q40 -40 62 10 l20 50" stroke="{ACCENT2}"/>
    <path d="M780 310 a60 60 0 1 0 0.1 0" stroke="{ACCENT2}"/>
    <path d="M930 430 q45 -140 90 0 q45 -140 90 0" stroke="{ACCENT2}"/>
  </g>
  <text x="90" y="590" font-family="ui-monospace, Menlo, monospace" font-size="26" fill="{INK}">runs in a browser tab · no server · Apache-2.0 · WER 1.3% held-out</text>
</svg>
"""

BANNER = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 400" width="1500" height="400" role="img" aria-label="zerre-tts banner">
  <rect width="1500" height="400" fill="{PAPER}"/>
  <text x="80" y="180" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="72" font-weight="700" fill="{INK}">zerre-tts</text>
  <text x="80" y="240" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="34" fill="{ACCENT}">8.5 MB · tarayıcıda · Türkçe TTS</text>
  <g fill="none" stroke="{ACCENT}" stroke-width="10" stroke-linecap="round" opacity="0.85">
    <path d="M700 260 q55 -160 110 0 q55 -160 110 0"/>
    <path d="M990 260 v-140 m0 70 q48 -48 74 12 l24 58"/>
    <path d="M1170 150 a70 70 0 1 0 0.1 0" stroke="{ACCENT2}"/>
  </g>
</svg>
"""


def main(argv=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in (("logo.svg", LOGO), ("og-image.svg", OG), ("banner.svg", BANNER)):
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
