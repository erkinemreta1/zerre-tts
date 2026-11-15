# zerre-tts

Türkçe, küçük ve tarayıcıda çalışan bir metin-konuşma modeli. Sıfırdan eğitilir;
hazır ağırlık kullanmaz. Mimarî [tiny-voca](https://github.com/n4ze3m/tiny-voca)
projesinden uyarlanmıştır (Apache-2.0), veri ve dil katmanı yenidir.

- ~4.5M parametre, 24 kHz, tek checkpoint.
- Fonem tabanlı: metin → eSpeak NG (`tr`) fonemleri → akustik model → kaynak-filtre vocoder.
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
