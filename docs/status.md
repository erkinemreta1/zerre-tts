# Durum

Son güncelleme: 2026-11-16

## Bitti

| Adım | Kanıt |
|---|---|
| Repo + CI (3 OS) | https://github.com/erkinemreta1/zerre-tts, Actions yeşil |
| Türkçe frontend | `lang/tr.py`, 62 token / 45 fonem, 34 test geçiyor |
| Taşınabilirlik | `filelock`, `ZERRE_ESPEAK` keşfi, yolsuz kaynak ağacı testi |
| Veri seti incelemesi | 48 kHz doğrulandı (kartlarda yazmıyordu) |
| Tam korpus | 8.650 klip, 8.0 sa MK + 8.01 sa NK = 16.01 sa |
| Akustik hazırlık | YIN + otokorelasyon ötümlülük, `feature_version: 3` |
| Hizalama | 24 kliplik probe: 0.843 / 0.884 (upstream 0.58-0.66) |
| ONNX yolu | 20.4 MB export, PyTorch ile korelasyon 0.9818 (gürültü tabanı 0.9820) |
| Web demosu | `web/public/` + 25.344 kelimelik sözlük (%99.98), Pages workflow |

## Sıradaki

1. Annotasyonun bitmesi (`data/tr/annotations/{train,validation,test}.jsonl`).
2. `scripts/package_training.py` ile bütünlük kontrolü + HF'ye yükleme.
3. Kiralık 4090'da eğitim (`docs/train.md`).
4. Export → `web/public/model.onnx` → Pages demosu canlı.

## Engeller

- **GPU**: bu makinede CUDA yok (Intel Arc 130V iGPU, torch `2.15.0+cpu`).
  Gerçek eğitim için kiralık 4090 gerekiyor (~7 $). Kullanıcı tarafında
  hesap/ödeme gereken tek adım.

## İlerlemeyi izleme

`docs/status.html` gerçek durumdan üretilen bir panodur (uydurma sayı yok).

```bash
uv run python scripts/status_report.py          # bir kez üret
uv run python scripts/status_report.py --open   # üret ve tarayıcıda aç
```

Sayfa 30 saniyede bir kendini yeniler; yeni sayıları görmek için betiği tekrar
çalıştırmak yeterli. Sürekli izlemek için:

```powershell
while ($true) { uv run python scripts/status_report.py; Start-Sleep 30 }
```

