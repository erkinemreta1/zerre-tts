# Durum

Son güncelleme: 2026-11-15

## Bitti

| Adım | Kanıt |
|---|---|
| Repo + CI (3 OS) | https://github.com/erkinemreta1/zerre-tts, Actions yeşil |
| Türkçe frontend | `lang/tr.py`, 62 token / 45 fonem, 40 cümlelik test seti geçiyor |
| Taşınabilirlik | `filelock`, `ZERRE_ESPEAK` keşfi, yolsuz kaynak ağacı testi |
| Veri seti incelemesi | 48 kHz doğrulandı (kartlarda yazmıyordu) |
| Tam korpus | 8.650 klip, 8.0 sa MK + 8.01 sa NK = 16.01 sa |
| Akustik hazırlık | 8.491 kabul / 159 red (val+test satırları) → `data/tr/features_full` |
| Hizalama | 100 kliplik pilot: 66 kabul, timing probe 0.606 (upstream 0.58-0.64) |
| ONNX yolu | 20.4 MB export, PyTorch ile korelasyon 0.9818 (gürültü tabanı 0.9820) |

## Şu an çalışıyor

Tam korpus annotasyonu: 8.650 klip, CPU, tahmini 3-4 saat.
Çıktı: `data/tr/annotations/{train,validation,test}.jsonl`.

## Sıradaki

1. Annotasyon kabul oranını ölç; %20'nin üzerindeyse QC eşiklerini gözden geçir
   (bkz. `docs/alignment.md`).
2. CPU'da 5-10 step'lik eğitim smoke'u: loss düşüyor mu, checkpoint yazılıyor mu.
3. Kiralık 4090'da gerçek koşu (`docs/train.md`).
4. Export + GitHub Pages demosu.

## Engeller

- **GPU**: bu makinede CUDA yok (Intel Arc 130V iGPU). Gerçek eğitim için kiralık
  4090 gerekiyor (~7 $). Bu, kullanıcı tarafında hesap/ödeme gerektiren tek adım.
- Whiper QC tam korpusta henüz koşmadı; pilot 20 klipte medyan WER 0.0, p90 0.182
  olduğu için v1'de kaynak metinler kullanılıyor.
