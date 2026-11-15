# Veri hattı

## Kaynaklar

| Ses | Veri seti | Klip (bildirilen) | Format |
|---|---|---|---|
| `MK` | [omersaidd/tts_mazlum_kiper_tur](https://huggingface.co/datasets/omersaidd/tts_mazlum_kiper_tur) | 9.643 | parquet, gömülü ses |
| `NK` | [omersaidd/tts_nisan_kumru_tur](https://huggingface.co/datasets/omersaidd/tts_nisan_kumru_tur) | 8.042 | parquet, gömülü ses |

Ölçülenler (2026-11-15, `scripts/inspect_datasets.py`):

- **Örnekleme oranı 48 kHz** (kartlarda yazmıyor). `zerre.prepare` 24 kHz ister, bu
  yüzden `scripts/build_parquet.py` `soxr` VHQ ile 48 → 24 kHz indirir.
- Süre: medyan 5.9-6.3 sn, aralık 1-32 sn. Metin: medyan 70-91 karakter.
- Parquet şerit boyutu: MK ~363 MB (23 şerit), NK ~380 MB (24 şerit) → toplam ~17 GB.

## Hattı

```
HF şeritleri
  └─ build_parquet.py   decode → 48→24 kHz → filtre → tekilleştir → split
        └─ data/tr/*.parquet       (upstream ile aynı kolonlar)
              └─ zerre.prepare     wave + log-mel + F0/voicing/energy
                    └─ data/tr/features/*.pt + train.jsonl
                          └─ zerre.annotate   (sonraki adım: hizalama)
```

## Filtreler (`build_parquet.py`)

| Kural | Değer |
|---|---|
| Süre | 1.0-15.0 sn |
| Metin uzunluğu | 15-300 karakter |
| Sessizlik | tepe < 0.01 veya RMS < 1e-4 → at |
| Kırpma | tepe > 0.999 → at |
| Tekilleştirme | (ses, normalize metin) çifti |
| Split | metin hash'i: %98 train, %1 validation, %1 test |

## Pilot koşu

`--shards 1 --max-hours 5`:

| | Klip | Saat |
|---|---:|---:|
| MK | 415 | 0.72 |
| NK | 329 | 0.67 |
| **Toplam** | **744** | **1.40** |

Reddedilen: süre 7, kırpma 4, metin uzunluğu 1. Çıktı: `data/tr/pilot.parquet` (129.7 MB).
`zerre.prepare` bu dosyadan 731 klibi kabul etti (13 red), örnek satır:
`wave (114688,)`, `mel (448, 80)`, `prosody (448, 3)`.

## Tam koşu

```bash
uv run python scripts/build_parquet.py --shards 0 --max-hours 20 \
  --output data/tr/full.parquet --report data/reports/full.json
uv run python -m zerre.prepare --inputs data/tr/full.parquet --split train --output data/tr/features
```

## Bilinen sorunlar

- **Metin-ses uyuşmazlığı**: metinler kitaptan geldiği için OCR/transkripsiyon
  hataları var (`Frankçe`, `daktili` gibi). Hizalama öncesi Whisper ile WER
  doğrulaması yapılmalı; eşik: ≤ %10 metni kullan, %10-25 Whisper metnini kullan,
  > %25 at.
- Kaynak klipler tek anlatıcı ve roman/öykü tonunda: soru/ünlem prosodisi zayıf
  kalabilir. Gerekirse `afkfatih/turkish-tts-combined-raw` içinden soru/ünlem
  klipleri eklenir.
