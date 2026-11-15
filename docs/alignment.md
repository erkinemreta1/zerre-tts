# Hizalama (annotation)

## Yöntem

Aynı çok dilli fonem tanıyıcı Türkçe için de kullanılıyor:
`facebook/wav2vec2-lv-60-espeak-cv-ft` (eSpeak fonem etiketleriyle CTC, 392 etiket).
Türkçe envanterin **44 foneminden 43'ü** bu sözlükte doğrudan var; iki istisna
dil paketinde eşleniyor:

| Envanter | Tanıyıcı etiketi | Neden |
|---|---|---|
| `g` | `ɡ` (U+0261) | Tanıyıcının eSpeak sürümü IPA ɡ yazıyor |
| `ɯː` | `ɯ` | Tanıyıcıda uzun ı etiketi yok; süre hizası için kısa etiket yeter |

`PhoneRecognizer` açılırken envanterdeki her fonemin bir etiketi olduğunu
doğrular (`align.label_fallback()`), eksik varsa hata verir.

## Gerçek korpusta çıkan iki sürpriz

744 kliplik pilot metin taramasında eSpeak NG 1.52 iki farklı yazım üretti:

| eSpeak çıktısı | Adet | Çözüm |
|---|---:|---|
| `ɡ` (U+0261) | 115 | `FALLBACK["ɡ"] = ("g",)` |
| `ɯɯ` (uzun ı, çift glif) | 132 | `FALLBACK["ɯɯ"] = ("ɯː",)` + `ɯː` envantere eklendi |

Bunlar düzeltilmeden 100 kliplik annotasyonda 23 klip `unknown_phoneme` ile
reddediliyordu. Düzeltmeden sonra aynı 100 klipte:

| | Önce | Sonra |
|---|---:|---:|
| Kabul | 51 | **66** |
| `unknown_phoneme` | 23 | 0 |

## Kalite probu (pilot, 100 klip, MK)

Sınırlarda ötümlülük dönüşü (İngilizce upstream değerleriyle karşılaştırma):

| Sınır | Türkçe `flip_found` | Upstream F2 | Upstream M2 |
|---|---:|---:|---:|
| ünlü → ötsüz ünsüz | 0.606 (642 sınır) | 0.638 | 0.580 |
| sürtünmeli → ünlü | 0.674 (242 sınır) | 0.663 | 0.659 |

Yani hizalama İngilizce orijinalle aynı aralıkta. `mean_offset_frames` -0.005,
`within_1_frame` 0.54.

## Kalan retler ve nedenleri

Pilot 100 klipte 34 ret: `weak_alignment` 26, `vowels_not_voiced` 10,
`low_confidence` 4. Bunlar upstream'den gelen QC eşikleri; İngilizce stüdyo
kayıtları için kalibre edilmiş. Türkçe için iki yol var:

1. Eşikleri olduğu gibi bırak, %34 reddi kabul et (16 saatten ~10.5 saat kalır).
2. `vowels_not_voiced` ve `min_token_probability` eşiklerini gevşet, red oranını
   %10-15'e indir; ama gürültülü hizalamaları riske at.

Karar için tam koşunun dağılımına bakılacak (`summary.json`).

## Komut

```bash
uv run python -m zerre.annotate \
  --features data/tr/features_full --output data/tr/annotations \
  --dataset-revision local --device cpu
```

GPU'da `--device cuda`; Mac'te `--device mps` (wav2vec2 MPS'te çalışır, daha
yavaş).
