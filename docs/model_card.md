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

# zerre-tts (tr-v1) — 8 MB, tarayıcıda çalışan Türkçe TTS

İki sesli, sıfırdan eğitilmiş Türkçe metin-konuşma modeli. **int8 ONNX hâli 8.1 MB**
ve tarayıcıda WebAssembly ile çalışır — sunucu, API, GPU gerekmez.

- 🤗 Demo: https://erkinemreta1.github.io/zerre-tts/
- 🧑‍💻 Kod ve eğitim tarifi: https://github.com/erkinemreta1/zerre-tts
- Mimarî [tiny-voca](https://github.com/n4ze3m/tiny-voca) uyarlaması (Apache-2.0).

## Neden bu

| | |
|---|---|
| Boyut | **8.1 MB int8** (fp32: 21.4 MB) |
| Parametre | 4.5M |
| Ses | 24 kHz, 2 ses (MK erkek, NK kadın) |
| Lisans | Apache-2.0 — ticari kullanım serbest |
| Çalıştığı yer | Tarayıcı (WASM), CPU, ONNX Runtime, PyTorch |
| Türkçe WER | **1.3%** (held-out 20 klip) · **12.7%** (8 yeni cümle) |

Quantization ölçümü (8 held-out klipte, Whisper large-v3):

| Varyant | WER | Dosya |
|---|---:|---:|
| fp32 | 15.3% | 21.38 MB |
| **int8** | **15.3%** | **8.11 MB** |

Yani 13 MB'ı **anlaşılırlıktan hiçbir şey kaybetmeden** attık. Aynı sette
filtre zinciri (90 Hz high-pass + 3 kHz presence + spektral temizlik) alt bant
enerjisini %7.3'ten %4.2'ye indiriyor.

## Dosyalar

| Dosya | Boyut | Ne için |
|---|---:|---|
| **`onnx/model-int8.onnx`** | **8.1 MB** | **Önerilen.** Tarayıcı, gömülü, dağıtım |
| `onnx/model.onnx` | 21.4 MB | fp32, referans/karşılaştırma |
| `onnx/model.json` | — | girdi-çıktı tanımı, ses listesi, sembol tablosu |
| `inference.pt` | 18.2 MB | PyTorch checkpoint (yeniden eğitim/ince ayar) |
| `samples/` | — | ham ve filtrelenmiş örnekler |
| `metrics/` | — | eğitim günlüğü, GPU doluluk raporu |

## Kullanım

### ONNX (int8)

Girdiler: `tokens` (int64 [1,N]), `stress` (int64 [1,N]), `voice` (int64 [1]),
`speed`, `pitch_shift`, `pitch_range` (float32 [1]). Çıktı: `audio` (float32).
Metin→fonem modelin dışında: eSpeak NG `tr` veya demoda kullanılan sözlük+kural.

### PyTorch

```bash
python -m zerre.infer --checkpoint inference.pt --lang tr \
  --text "Merhaba dünya." --voice MK --output hello.wav
```

## Eğitim

İki açık Türkçe sesli kitap korpusu (`omersaidd/tts_mazlum_kiper_tur`,
`omersaidd/tts_nisan_kumru_tur`): 16 saat, 48→24 kHz indirildi, eSpeak fonem
tanıyıcıyla hizalandı, 6.347 klip (11.93 saat) kabul edildi. 65.000 step,
tek L4'te ~6.5 saat, ~7 $.

## Sınırlar

- İki sabit ses; klonlama ve streaming yok.
- Okuma tonu ağırlıklı; soru/ünlem prosodisi sınırlı.
- Yabancı isimler ve kısaltmalar sık yanlış okunur.
- WER ölçümleri küçük setlerde (20 klip / 8 cümle); mutlak değil, sanity check.

## Atıf

```bibtex
@software{zerre_tts_2026,
  title  = {zerre-tts: a 4.5M-parameter Turkish two-voice text-to-speech model},
  year   = {2026},
  url    = {https://github.com/erkinemreta1/zerre-tts}
}
```

Mimarî ve eğitim tarifi: Muhammed Nazeem, *tiny-voca* (Apache-2.0).
