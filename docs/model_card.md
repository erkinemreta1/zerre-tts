---
language:
  - tr
license: apache-2.0
tags:
  - text-to-speech
  - tts
  - turkish
  - onnx
  - int8
  - browser
library_name: pytorch
pipeline_tag: text-to-speech
---

# zerre-tts (tr-v1) — an 8.5 MB Turkish TTS that runs in a browser tab

A two-voice Turkish text-to-speech model, trained from scratch. The **int8 ONNX
build is 8.5 MB** and runs in the browser via WebAssembly — no server, no API,
no GPU.

- 🎧 Demo: https://erkinemreta1.github.io/zerre-tts/
- 🧑‍💻 Code and training recipe: https://github.com/erkinemreta1/zerre-tts
- Architecture adapted from [tiny-voca](https://github.com/n4ze3m/tiny-voca) (Apache-2.0).

Türkçe: iki sesli, sıfırdan eğitilmiş Türkçe metin-konuşma modeli. Tarayıcıda
çalışır, sunucu gerekmez; ölçümler ve kıyaslar aşağıdadır.

## Why this one

| | |
|---|---|
| Size | **8.5 MB int8** (fp32: 21.4 MB) |
| Parameters | 4.5M |
| Voices | 24 kHz, two voices (SES-A, SES-B) |
| Licence | Apache-2.0 — commercial use allowed |
| Runs on | Browser (WASM), CPU, ONNX Runtime, PyTorch |
| Turkish WER | **1.3%** (20 held-out clips) · **12.7%** (8 new sentences) |

Quantisation, measured on 8 held-out clips with Whisper large-v3:

| Variant | WER | File |
|---|---:|---:|
| fp32 | 14.0% | 21.38 MB |
| **int8** | **15.3%** | **8.55 MB** |

13 MB smaller for no measurable cost in intelligibility. On the same clips, the
post-processing chain (90 Hz high-pass, +2.5 dB presence at 3 kHz, spectral
subtraction) halves the below-120 Hz energy that read as boom.

## Where it stands

Among browser-capable Turkish TTS models, this one leads on size per unit of
quality (eval sets differ; read the ratio, not the absolutes):

| Model | Params | Deploy size | Turkish WER | MB/WER | Licence | Browser | Voices |
|---|---:|---:|---:|---:|---|---|---|
| **zerre-tts (int8)** | **4.5M** | **8.5 MB** | **1.3%** | **6.5** | **Apache-2.0** | ✅ | **2** |
| okur-tts | 9.8M | ~23 MB | 1.81% | 12.8 | Apache-2.0 | ✅ | 1 |
| Piper `tr_TR-dfki` | ~16M | 63 MB | 3.14% | 20.1 | CC BY-NC-SA | ✅ | 1 |
| Meta MMS-TTS | 36M | ~145 MB | 5.74% | 25.3 | CC BY-NC | ✅ | 1 |
| sanoTTS | 0.5-2.3M | ~2.8 MB | 30.3% | 0.09 | GPL-3.0 | ✅ | many |
| Kokoro-82M | 82M | ~330 MB | no Turkish | — | Apache-2.0 | ✅ | many |
| XTTS-v2 | 467M | ~2 GB | — | — | CPML (NC) | ❌ | cloning |

Smallest deploy, best MB/WER, commercial-friendly licence, two voices. It is
not the most natural: 1-2 GB server-class models still sound better.

## Files

| File | Size | Use |
|---|---:|---|
| **`onnx/model-int8.onnx`** | **8.5 MB** | **Recommended.** Browser, embedded, distribution |
| `onnx/model.onnx` | 21.4 MB | fp32, reference/comparison |
| `onnx/model.json` | — | input/output spec, voice list, symbol table |
| `inference.pt` | 18.2 MB | PyTorch checkpoint (retraining/fine-tuning) |
| `samples/` | — | raw and post-processed examples |
| `metrics/` | — | training journal, GPU utilisation report |

## Usage

### ONNX (int8)

Inputs: `tokens` (int64 [1,N]), `stress` (int64 [1,N]), `voice` (int64 [1]),
`speed`, `pitch_shift`, `pitch_range` (float32 [1]). Output: `audio` (float32).
Text→phoneme is outside the graph: eSpeak NG `tr`, or the lexicon-plus-rules
phonemizer used by the demo.

### PyTorch

```bash
python -m zerre.infer --checkpoint inference.pt --lang tr \
  --text "Merhaba dünya." --voice SES-A --output hello.wav
```

One sentence per call; split longer text at sentence boundaries.

## Training

Openly licensed Turkish audiobook corpora (identities and the licence caveat:
`docs/data_sources.md` in the repository): 16 h, downsampled 48→24 kHz, aligned
with an eSpeak phoneme recogniser; 6,347 clips (11.93 h) accepted. 65,000 steps,
~6.5 h on a single L4, ~$7.

## Voice and data note

The voices were trained from openly licensed recordings; **no cloning was done**,
no speaker identity is claimed and there is no connection to the speakers. The
voices are published under neutral labels (SES-A, SES-B). Details and caveats:
`docs/data_sources.md`. Legal review for commercial use is the user's
responsibility.

## Limits

- Two fixed voices (SES-A, SES-B); no cloning, no streaming.
- Read-speech prosody; questions and exclamations are weak.
- Foreign names and abbreviations are often wrong.
- WER was measured on small sets (20 clips / 8 sentences): a sanity check, not
  a benchmark.

## Citation

```bibtex
@software{zerre_tts_2026,
  title  = {zerre-tts: a 4.5M-parameter Turkish two-voice text-to-speech model},
  year   = {2026},
  url    = {https://github.com/erkinemreta1/zerre-tts}
}
```

Architecture and training recipe: Muhammed Nazeem, *tiny-voca* (Apache-2.0).
