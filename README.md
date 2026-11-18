# zerre-tts

A small Turkish text-to-speech model that runs in a browser tab. Trained from
scratch — no pretrained weights anywhere in the pipeline. The architecture is
adapted from [tiny-voca](https://github.com/n4ze3m/tiny-voca) (Apache-2.0); the
data, the language layer and everything around them are new.

**Demo: https://erkinemreta1.github.io/zerre-tts/** ·
**Weights: https://huggingface.co/erkint/zerre-tts**

- **8.5 MB int8 ONNX** (fp32: 21.4 MB) — quantisation changes nothing measurable.
- ~4.5M parameters, 24 kHz, two voices (`SES-A`, `SES-B`).
- Evaluation: **1.3% WER** on held-out clips, **12.7%** on new sentences
  (Whisper large-v3, Turkish). Small sets — a sanity check, not a benchmark.
- Phoneme-based: text → eSpeak NG (`tr`) phonemes → acoustic model →
  source-filter vocoder. In the browser, a 25k-word lexicon plus rules.
- Language-pack design: add a language under `src/zerre/lang/`, the core stays
  untouched. A Kurdish (Kurmanji) pack is planned.
- No server, no API key, no upload: the model downloads once and the audio is
  synthesised on the device.

Türkçe özet: yukarıdaki her şey — küçük, tarayıcıda çalışan, sıfırdan eğitilmiş
Türkçe TTS. Ayrıntılı sayılar ve kıyas tablosu aşağıda.

## Comparison

Among browser-capable Turkish TTS models, this one leads on size per unit of
quality. Eval sets differ, so read the ratio, not the absolutes.

| Model | Params | Deploy size | Turkish WER | MB/WER | License | Browser | Voices |
|---|---:|---:|---:|---:|---|---|---|
| **zerre-tts (int8)** | **4.5M** | **8.5 MB** | **1.3%** | **6.5** | **Apache-2.0** | ✅ | **2** |
| okur-tts | 9.8M | ~23 MB | 1.81% | 12.8 | Apache-2.0 | ✅ | 1 |
| Piper `tr_TR-dfki` | ~16M | 63 MB | 3.14% | 20.1 | CC BY-NC-SA | ✅ | 1 |
| Meta MMS-TTS | 36M | ~145 MB | 5.74% | 25.3 | CC BY-NC | ✅ | 1 |
| sanoTTS | 0.5-2.3M | ~2.8 MB | 30.3% | 0.09 | GPL-3.0 | ✅ | many |
| Kokoro-82M | 82M | ~330 MB | no Turkish | — | Apache-2.0 | ✅ | many |
| XTTS-v2 | 467M | ~2 GB | — | — | CPML (NC) | ❌ | cloning |

What it wins on: smallest deploy, best MB/WER, commercial-friendly licence,
two voices. What it is not: the most natural — 1-2 GB server-class models still
sound better.

## Quick start

```bash
git clone https://github.com/erkinemreta1/zerre-tts.git
cd zerre-tts
uv venv --python 3.11 && uv pip install -e ".[dev]"

# needs eSpeak NG: winget install eSpeak-NG.eSpeak-NG
#                 (brew install espeak-ng / apt install espeak-ng)

uv run pytest -q                                   # 39 tests
uv run python -c "from zerre.text import phonemize; \
  print(' '.join(phonemize('Merhaba dünya!').symbols))"
```

Synthesis (needs a checkpoint from the HF repo):

```bash
uv run python -m zerre.infer --checkpoint inference.pt --lang tr \
  --text "Merhaba dünya." --voice SES-A --output hello.wav
```

In the browser, the same model runs as int8 ONNX with a lexicon-based
phonemizer — see `web/`.

## Repository layout

```
src/zerre/          model, training, front end, alignment, export
src/zerre/lang/     language packs (tr; ku planned)
scripts/            data pipeline, evaluation, publishing, demo tooling
tests/              frontend, config, portability, prosody
docs/               model card, data sources, runbooks, launch copy
web/                static browser demo (GitHub Pages)
```

Data, weights and runs are never committed; see `docs/data_sources.md` for
provenance and licence caveats.

## Documentation

- `docs/model_card.md` — the HF model card, with measurements and limits
- `docs/data.md` / `docs/alignment.md` — corpus and alignment details
- `docs/train.md` / `docs/modal.md` — training runbooks (rented GPU / Modal)
- `docs/launch.md` — press copy and prepared answers
- `docs/mediakit.md` — media kit: one-liners, boilerplate, assets, facts
- `docs/media/` — logo, og-image, banner (SVG)
- `docs/media/` — logo, og-image and banner (SVG)

## Media kit

One-liners, EN/TR boilerplate, verified facts and assets live in
[`docs/mediakit.md`](docs/mediakit.md); logos are in [`docs/media/`](docs/media/).

## Citation

```bibtex
@software{zerre_tts_2026,
  title  = {zerre-tts: a 4.5M-parameter Turkish two-voice text-to-speech model},
  year   = {2026},
  url    = {https://github.com/erkinemreta1/zerre-tts}
}
```

Architecture and training recipe: Muhammed Nazeem, *tiny-voca* (Apache-2.0).

## Licence

Apache-2.0 for code and weights. Voice provenance and the licence caveat for the
source recordings: `NOTICE` and `docs/data_sources.md`.
