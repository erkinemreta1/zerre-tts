"""Render the zerre-tts logo and og-image as SVG (text-first, no binary deps).

    uv run python scripts/make_media.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"

INK = "#12141a"
PAPER = "#fbfbfa"
ACCENT = "#0b6b5b"
ACCENT2 = "#5ad2b8"

LOGO = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 220" width="960" height="220" role="img" aria-label="zerre-tts">
  <rect width="960" height="220" fill="{PAPER}"/>
  <g fill="none" stroke="{ACCENT}" stroke-width="14" stroke-linecap="round">
    <path d="M60 150 q30 -90 60 0 q30 -90 60 0"/>
    <path d="M220 150 l40 -60 l-40 0 l40 60" stroke="{ACCENT2}"/>
    <path d="M320 90 a32 32 0 1 1 0.1 0" stroke="{ACCENT2}"/>
    <path d="M400 90 v60 m0 -30 h44 m0 -30 v60"/>
    <path d="M490 90 v60 h40"/>
    <path d="M570 150 v-60 q30 -12 44 0 v60"/>
    <path d="M660 90 v60 m0 -30 q20 -20 34 4 l10 26"/>
    <path d="M740 90 v60 m0 -46 a26 26 0 1 1 20 44 l-20 2"/>
    <path d="M830 90 v60 l30 -28 l30 28 v-60" stroke="{ACCENT2}"/>
  </g>
  <text x="60" y="40" font-family="ui-monospace, 'Cascadia Code', Menlo, monospace" font-size="28" fill="{INK}">zerre-tts — 8.5 MB Türkçe TTS</text>
</svg>
"""

OG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630" role="img" aria-label="zerre-tts">
  <rect width="1200" height="630" fill="{PAPER}"/>
  <rect x="0" y="560" width="1200" height="70" fill="{ACCENT}" opacity="0.12"/>
  <text x="80" y="150" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="84" font-weight="700" fill="{INK}">zerre-tts</text>
  <text x="80" y="230" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="40" fill="{ACCENT}">8.5 MB Turkish TTS, trained from scratch</text>
  <g fill="none" stroke="{ACCENT}" stroke-width="10" stroke-linecap="round" opacity="0.9">
    <path d="M80 380 q40 -120 80 0 q40 -120 80 0"/>
    <path d="M300 380 v-100 m0 50 q30 -30 50 6 l14 44"/>
    <path d="M460 380 v-100 l40 56 l40 -56 v100" stroke="{ACCENT2}"/>
    <path d="M640 280 a52 52 0 1 1 0.1 0" stroke="{ACCENT2}"/>
    <path d="M770 380 v-100 m0 30 q36 -20 60 8 l16 62"/>
    <path d="M920 380 l50 -76 l-50 0 l50 76" stroke="{ACCENT2}"/>
  </g>
  <text x="80" y="600" font-family="ui-monospace, Menlo, monospace" font-size="26" fill="{INK}">runs in a browser tab · no server · Apache-2.0 · werre 1.3% held-out</text>
</svg>
"""

BANNER = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 400" width="1500" height="400" role="img" aria-label="zerre-tts banner">
  <rect width="1500" height="400" fill="{PAPER}"/>
  <g fill="none" stroke="{ACCENT}" stroke-width="9" stroke-linecap="round" opacity="0.85">
    <path d="M70 260 q45 -130 90 0 q45 -130 90 0"/>
    <path d="M310 260 v-110 m0 55 q34 -34 56 8 l18 47"/>
    <path d="M470 260 v-110 l44 62 l44 -62 v110" stroke="{ACCENT2}"/>
    <path d="M650 150 a56 56 0 1 1 0.1 0" stroke="{ACCENT2}"/>
    <path d="M790 260 v-110 m0 33 q40 -22 66 9 l18 68"/>
    <path d="M950 260 l55 -84 l-55 0 l55 84" stroke="{ACCENT2}"/>
  </g>
  <text x="1120" y="200" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="56" font-weight="700" fill="{INK}" text-anchor="middle">zerre-tts</text>
  <text x="1120" y="250" font-family="system-ui, 'Segoe UI', Roboto, sans-serif" font-size="28" fill="{ACCENT}" text-anchor="middle">8.5 MB · tarayıcıda · Türkçe</text>
</svg>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in (("logo.svg", LOGO), ("og-image.svg", OG), ("banner.svg", BANNER)):
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
