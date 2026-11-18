# zerre-tts media kit

Everything needed to write about, share or embed the project. Updated: 2026-11-18.

## Links

| What | URL |
|---|---|
| Demo | https://erkinemreta1.github.io/zerre-tts/ |
| Weights (HF) | https://huggingface.co/erkint/zerre-tts |
| Code (GitHub) | https://github.com/erkinemreta1/zerre-tts |
| Model card | https://huggingface.co/erkint/zerre-tts (README) |

## One-liners

- **EN:** An 8.5 MB Turkish text-to-speech model that runs entirely in a browser
  tab — trained from scratch, Apache-2.0.
- **TR:** Tarayıcıda çalışan, 8.5 MB'lık, sıfırdan eğitilmiş Türkçe konuşma
  sentezleyici (Apache-2.0).
- **Shortest:** 8.5 MB. Turkish TTS. In your browser.

## Boilerplate (EN)

zerre-tts is a two-voice Turkish text-to-speech model with 4.5 million
parameters, trained from scratch on openly licensed audiobook recordings. The
int8 ONNX build is 8.5 MB and synthesises speech inside the browser with
WebAssembly — no server, no API key, and the audio never leaves the device. On
held-out clips it reaches 1.3% word error rate (Whisper large-v3); on sentences
it has never seen, 12.7%. Training took about seven hours on a single rented
GPU and cost roughly seven dollars. Code, weights and the full training recipe
are open under Apache-2.0.

## Boilerplate (TR)

zerre-tts, açık lisanslı sesli kitap kayıtlarıyla sıfırdan eğitilmiş, 4.5 milyon
parametreli, iki sesli bir Türkçe metin-konuşma modeli. int8 ONNX hâli 8.5 MB ve
konuşmayı tarayıcının içinde (WebAssembly) üretir — sunucu, API anahtarı yoktur,
ses cihazdan çıkmaz. Held-out kliplerde kelime hata oranı %1.3 (Whisper
large-v3), hiç görmediği cümlelerde %12.7. Eğitim tek kiralık GPU'da yaklaşık
yedi saat sürdü ve kabaca yedi dolara mal oldu. Kod, ağırlıklar ve eğitim
tarifi Apache-2.0 ile açık.

## Facts (do not misquote)

| Claim | Value | Caveat |
|---|---|---|
| Model size | 8.5 MB int8 ONNX | fp32 is 21.4 MB; the ORT WASM runtime downloads ~10 MB more |
| Parameters | 4,534,603 | inference-only |
| Voices | 2 (SES-A, SES-B) | neutral labels, no cloning |
| Sample rate | 24 kHz | |
| WER, held-out | 1.3% | 20 clips, same domain |
| WER, new sentences | 12.7% | 8 sentences × 2 voices |
| Training | 65,000 steps, ~6.5 h, 1× L4 | ~$7 |
| Data | 11.93 h accepted of 16 h | openly licensed audiobooks; see `docs/data_sources.md` |
| Licence | Apache-2.0 (code + weights) | voice-provenance caveat in `docs/data_sources.md` |

## Numbers to compare (as of 2026-11)

| Model | Deploy size | Turkish WER | Licence |
|---|---:|---:|---|
| zerre-tts int8 | 8.5 MB | 1.3%* | Apache-2.0 |
| okur-tts | ~23 MB | 1.81% | Apache-2.0 |
| Piper tr_TR-dfki | 63 MB | 3.14% | CC BY-NC-SA |
| Meta MMS-TTS | ~145 MB | 5.74% | CC BY-NC |
| XTTS-v2 | ~2 GB | — | non-commercial |

\* 20 clips, okur's number is from a 495-sentence set — different protocols,
ratio not ranking.

## Assets

| File | Use |
|---|---|
| `docs/media/logo.svg` | wordmark + waveform, light background |
| `docs/media/og-image.svg` | social preview, 1200×630 |
| `docs/media/banner.svg` | repo/README banner, 1500×400 |

Colours: ink `#12141a`, paper `#fbfbfa`, accent `#0b6b5b`, accent-2 `#5ad2b8`.
Type: system sans for headlines, monospace for annotations.

Audio examples for embedding: `samples/` in the HF repo (raw and
post-processed). For a 20-second demo, point a screen recorder at
https://erkinemreta1.github.io/zerre-tts/ and type.

## What to say / what not to say

Say: "trained from scratch", "runs in the browser", "8.5 MB",
"intelligibility-first", "Apache-2.0".

Avoid: "most natural Turkish TTS" (it is not), "voice cloning" (there is none),
"perfectly licensed voices" (provenance is best-effort; see
`docs/data_sources.md`). The WER numbers come from small sets — keep the
asterisk when quoting.

## Contact

Issues and questions: https://github.com/erkinemreta1/zerre-tts/issues
