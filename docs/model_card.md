---
language:
  - tr
license: apache-2.0
tags:
  - text-to-speech
  - tts
  - turkish
  - onnx
  - browser
library_name: pytorch
pipeline_tag: text-to-speech
---

# zerre-tts (tr-v1)

Türkçe, **4.5M parametreli**, sıfırdan eğitilmiş iki sesli bir metin-konuşma modeli.
Ağırlıklar **20.4 MB** (ONNX); tarayıcıda WebAssembly ile çalışır, sunucu gerekmez.

- Demo: https://erkinemreta1.github.io/zerre-tts/
- Kod ve eğitim tarifi: https://github.com/erkinemreta1/zerre-tts
- Mimarî [tiny-voca](https://github.com/n4ze3m/tiny-voca) projesinden uyarlanmıştır (Apache-2.0).

## Ölçümler

Whisper `large-v3` (Türkçe) ile kelime hata oranı:

| Set | Klip | WER |
|---|---:|---:|
| Test (aynı domain, held-out) | 20 | **1.3%** |
| Kayıtların kendisi (referans) | 20 | 0.0% |
| Yeni cümleler (hiçbir split'te yok) | 8 × 2 ses | **12.7%** |

Süre oranı 0.95, mel DTW 0.69, üretilen perde dağılımı kayıtla uyumlu.

> Bu setler küçük; mutlak değil, sanity check olarak okunmalı.

## Boyut karşılaştırması (Türkçe)

| Model | Parametre | Dağıtım boyutu | Türkçe WER | Lisans |
|---|---:|---:|---:|---|
| **zerre-tts (bu)** | **4.5M** | **20.4 MB** | **1.3%** (20 klip) | Apache-2.0 |
| okur-tts | 9.8M | ~23 MB ONNX | 1.81% (495 cümle) | Apache-2.0 |
| Piper tr_TR-dfki | ~16M | ~63 MB | 3.14% | CC BY-NC-SA |
| Meta MMS-TTS | 36M | ~145 MB | 5.74% | CC BY-NC |
| XTTS-v2 | 467M | ~2 GB | — | CPML (NC) |

## Kullanım (Python)

```bash
pip install zerre-tts
git clone https://github.com/erkinemreta1/zerre-tts
```

```bash
python -m zerre.infer --checkpoint inference.pt --lang tr \
  --text "Merhaba dünya." --voice MK --output hello.wav
```

Sesler: `MK` (erkek), `NK` (kadın). Girdi bir cümle olmalı; uzun metni bölmek çağıranın işi.

## Kullanım (ONNX / tarayıcı)

Girdiler: `tokens` (int64 [1,N]), `stress` (int64 [1,N]), `voice` (int64 [1]),
`speed`, `pitch_shift`, `pitch_range` (float32 [1]). Çıktı: `audio` (float32).
Metin→fonem adımı modelin dışında: eSpeak NG `tr` veya demoda kullanılan
sözlük + kural tabanlı dönüştürücü.

## Eğitim verisi

İki açık Türkçe sesli kitap korpusu (`omersaidd/tts_mazlum_kiper_tur`,
`omersaidd/tts_nisan_kumru_tur`) → 16 saat, 48 kHz'den 24 kHz'e indirildi.
Fonem hizalaması eSpeak fonem tanıyıcı ile yapıldı; 6.347 klip (11.93 saat)
kabul edildi. 65.000 step, tek L4'te ~6.5 saat, ~7 $.

## Sınırlar

- İki sabit ses, klonlama yok, akış (streaming) yok.
- Eğitim okuma tonunda; soru/ünlem prosodisi sınırlı.
- Yabancı isimler ve kısaltmalar sık yanlış okunur.
- Kaynak metinlerdeki OCR hataları az da olsa etiketlere yansımış olabilir.

## Atıf

```bibtex
@software{zerre_tts_2026,
  title  = {zerre-tts: a 4.5M-parameter Turkish two-voice text-to-speech model},
  year   = {2026},
  url    = {https://github.com/erkinemreta1/zerre-tts}
}
```

Mimarî ve eğitim tarifi: Muhammed Nazeem, *tiny-voca* (Apache-2.0).
