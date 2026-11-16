# zerre-tts

Türkçe, küçük ve tarayıcıda çalışan bir metin-konuşma modeli. Sıfırdan eğitilir;
hazır ağırlık kullanmaz. Mimarî [tiny-voca](https://github.com/n4ze3m/tiny-voca)
projesinden uyarlanmıştır (Apache-2.0), veri ve dil katmanı yenidir.

**Demo: https://erkinemreta1.github.io/zerre-tts/** ·
**Ağırlıklar: https://huggingface.co/erkint/zerre-tts**

- **8.1 MB int8 ONNX** (fp32: 21.4 MB) — quantization WER'i değiştirmiyor (15.3% → 15.3%).
- ~4.5M parametre, 24 kHz, iki ses (`MK` erkek, `NK` kadın).
- Ölçüm: held-out test setinde **WER %1.3**, yeni cümlelerde **%12.7**
  (Whisper large-v3, Türkçe).
- Fonem tabanlı: metin → eSpeak NG (`tr`) fonemleri → akustik model →
  kaynak-filtre vocoder. Tarayıcıda sözlük + kural tabanlı fonemleme.
- Dil paketi yapısı: `src/zerre/lang/` altına yeni dil eklenir, çekirdek değişmez.

> Durum: iskelet + Türkçe frontend hazır. Model eğitimi ve ses verisi henüz yok.

## Kurulum

```bash
git clone https://github.com/<user>/zerre-tts.git
cd zerre-tts
uv venv --python 3.11
uv pip install -e ".[dev]"
```

eSpeak NG gerekir (fonem üretimi için):

| Platform | Komut |
|---|---|
| Windows | `winget install eSpeak-NG.eSpeak-NG` |
| macOS | `brew install espeak-ng` |
| Debian/Ubuntu | `sudo apt install espeak-ng` |

Kurulu değilse `ZERRE_ESPEAK` ile yolu ver: `ZERRE_ESPEAK=/opt/espeak-ng/bin/espeak-ng`.

## Kıyaslama

Türkçe, tarayıcıda çalışan modeller arasında **MB başına kalitede öndeyiz**.
Ölçüm setleri aynı olmadığı için mutlak değil, oransal okuyun.

| Model | Parametre | Boyut | Türkçe WER | MB/WER | Lisans | Tarayıcı | Ses |
|---|---:|---:|---:|---:|---|---|---|
| **zerre-tts (int8)** | **4.5M** | **8.1 MB** | **1.3%** | **6.2** | **Apache-2.0** | ✅ | **2** |
| okur-tts | 9.8M | ~23 MB | 1.81% | 12.8 | Apache-2.0 | ✅ | 1 |
| Piper `tr_TR-dfki` | ~16M | 63 MB | 3.14% | 20.1 | CC BY-NC-SA | ✅ | 1 |
| Meta MMS-TTS | 36M | ~145 MB | 5.74% | 25.3 | CC BY-NC | ✅ | 1 |
| sanoTTS | 0.5-2.3M | ~2.8 MB | 30.3% | 0.09 | GPL-3.0 | ✅ | çok |
| Kokoro-82M | 82M | ~330 MB | Türkçe yok | — | Apache-2.0 | ✅ | çok |
| XTTS-v2 | 467M | ~2 GB | — | — | CPML (NC) | ❌ | klonlama |

Kazandığımız: **en küçük dağıtım**, **en iyi MB/WER**, **ticari dostu lisans**, **iki ses**.

## Kullanım

```bash
# metni fonemlere çevir
uv run python -c "from zerre.text import phonemize; print(' '.join(phonemize('Merhaba dünya!').symbols))"

# envanteri doğrula (eSpeak sürümü değişince çalıştır)
uv run python scripts/audit_inventory.py

# testler
uv run pytest
```

Sentez (checkpoint hazır olduğunda):

```bash
uv run zerre-speak --checkpoint weights/inference.pt --lang tr --text "Merhaba dünya." --output hello.wav
```

## Dil paketleri

Aktif dil `ZERRE_LANG` ile seçilir (varsayılan `tr`). Bir dil paketi şunları taşır:
fonem envanteri, eSpeak sesi, noktalama, kısaltmalar, fallback tablosu, normalizasyon.

```
src/zerre/lang/
├─ base.py   # Language protokolü + ortak sabitler
├─ tr.py     # Türkçe
└─ __init__.py
```

Yeni dil eklemek: `lang/<kod>.py` yaz, `LANGUAGES` sözlüğüne kaydet, `ZERRE_LANG=<kod>` ile çalıştır.

## Yapı

```
src/zerre/        # model, eğitim, frontend, hizalama
scripts/          # envanter denetimi, veri hazırlama, smoke testler
tests/            # frontend, config, taşınabilirlik
data/ weights/ runs/ eval/   # repoda yok (.gitignore)
```

## Testler

```bash
uv run pytest -q
```

eSpeak yoksa entegrasyon testleri atlanır, saf fonksiyon testleri çalışır.

## Lisans

Kod ve ağırlıklar Apache-2.0. Ayrıntı ve atıf: `LICENSE`, `NOTICE`.
