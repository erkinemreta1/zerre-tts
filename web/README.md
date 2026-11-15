# web

Tarayıcı demosu. Statik dosyalar: `index.html`, `worker.js`, `lexicon-tr.json`,
`model.json` ve `model.onnx`.

## Gerekli dosyalar

| Dosya | Nereden |
|---|---|
| `lexicon-tr.json` | `uv run python scripts/build_lexicon.py --output web/public/lexicon-tr.json` |
| `model.json`, `model.onnx` | `uv run python -m zerre.export --checkpoint runs/<ad>/inference.pt --output web/public/model.onnx` |

`model.onnx` depoda tutulmaz (20 MB); Hugging Face'ten çekilir.

## Yerel çalıştırma

```bash
cd web/public
python -m http.server 8000
# http://localhost:8000
```

`file://` ile açılmaz: worker, `fetch` ve WASM için HTTP gerekir.

## Nasıl çalışıyor

- Fonemler `lexicon-tr.json`'dan gelir (eSpeak NG `tr` ile üretilmiş, eğitimle
  aynı envanter). Sözlükte olmayan kelimeler için `worker.js` içindeki kural
  tabanlı dönüştürücü devreye girer.
- Model ONNX Runtime Web (wasm) ile worker thread'de koşar, cümle cümle sentezler
  ve her cümleyi hazır olduğunda çalar.
- Ses hiçbir sunucuya gitmez.

## GitHub Pages

`.github/workflows/pages.yml`, `web/public` klasörünü yayınlar. `model.onnx`
yoksa iş akışı `HF_ASSETS` değişkenindeki URL'den indirir.
