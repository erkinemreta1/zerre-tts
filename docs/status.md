# Durum

Son güncelleme: 2026-11-16 · **eğitim Modal'da sürüyor (`tr-v1`)**

| | |
|---|---|
| Veri | 6.347 klip · 11.93 sa (Modal volume, T4'te hizalandı: 568 sn) |
| Eğitim | 65.000 step hedefi, 6 saatlik dilim, L4 |
| Ölçülen | GAN sonrası 2.12 step/s · GPU %77 · 186 ses-sn/sn |
| Maliyet | veri işi ~0.10 $ + probe'lar ~0.10 $ + eğitim ~4.8 $ |

Örnek almak için (GPU sırası beklemeden, CPU'da):

```bash
uv run python scripts/taste.py                 # iki ses, son checkpoint
uv run python scripts/taste.py --voice NK --text "İstediğin cümle."
```

Dosya adı checkpoint'in gerçek adımını taşır: `samples/taste-step07000-mk.wav`.

Kaldığı yer: veri ve hizalama tamam, eğitim başladı. Devam/izleme komutları
`docs/modal.md`'de.

## Bitti

| Adım | Kanıt |
|---|---|
| Repo + CI (5 iş: 3 OS + package + onnx) | https://github.com/erkinemreta1/zerre-tts, yeşil |
| Türkçe frontend | `lang/tr.py`, 62 token / 45 fonem, 34 test |
| Taşınabilirlik | `filelock`, `ZERRE_ESPEAK` keşfi, yolsuz kaynak ağacı testi |
| Korpus | 8.650 klip · MK 8.0 sa + NK 8.0 sa (48 kHz kaynaktan 24 kHz) |
| Akustik özellikler | 8.650 `.pt`, YIN + otokorelasyon ötümlülük |
| Hizalama | **6.346 kabul / 8.491 (%74.7)**, 11.93 saat |
| Onay/test setleri | validation 63 klip, test 54 klip |
| Paket bütünlüğü | `package_training.py`: eksik dosya 0, parmak izi uyuşmazlığı 0 |
| CPU eğitim smoke'u | 6 step, loss 56.8 → 43.6, 13-17x realtime |
| ONNX yolu | 20.4 MB export, PyTorch ile korelasyon 0.9818 (taban 0.9820) |
| Web demosu | 25.344 kelimelik sözlük, eSpeak ile token oranı 0.9975 |

## Hizalama sonuçları

| Split | Kabul | Red | Saat | Ses |
|---|---:|---:|---:|---|
| train | 6.346 | 2.145 | 11.93 | MK 5.75 · NK 6.17 |
| validation | 63 | 22 | 0.12 | MK 0.06 · NK 0.05 |
| test | 54 | 20 | 0.10 | MK 0.05 · NK 0.05 |

Red gerekçeleri (train): `weak_alignment` 1.771, `low_confidence` 786,
`vowels_not_voiced` 160, `overlong_phone` 85.

Kalite probu (tüm korpus):

| Ses | ünlü→ötsüz | sürtünmeli→ünlü |
|---|---:|---:|
| MK | 0.874 | 0.933 |
| NK | 0.931 | 0.983 |

Upstream İngilizce referansı 0.58-0.66; Türkçe hizalaması daha keskin.

## Sıradaki

1. `hf upload` ile `annotations` + `features_full` paketini yükle.
2. Kiralık 4090'da eğitim (`docs/train.md`), 55-65k step.
3. Export → `web/public/model.onnx` → Pages demosu canlı.

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
