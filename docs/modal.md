# Modal'da eğitim

Modal, sunucusuz GPU kiralar; 4090 aramana gerek yok. Bu repo için hazır dosya:
`modal_train.py`.

## Tek seferlik kurulum

```bash
uv pip install modal
modal token new          # tarayıcı açılır, hesabınla onaylarsın
modal volume create zerre-tts
```

## Veriyi yükle (6.9 GB, bir kez)

```bash
modal volume put zerre-tts data/tr/annotations /annotations
modal volume put zerre-tts data/tr/features_full /features
```

Alternatif: veriyi Hugging Face'e yükleyip konteynerde indirmek (veri merkezi
bandı daha hızlı). O durumda `modal_train.py` içindeki `train()` başına bir
`hf_hub_download` eklenir.

## Eğitim

```bash
# tek komut, bitene kadar bekler
modal run modal_train.py --steps 65000 --hours 11
```

- `--hours 11` → `--max-seconds 39600`: süre dolunca checkpoint yazıp çıkar.
- Aynı komutu tekrar çalıştırırsan `last.pt` varsa **kaldığı yerden devam eder**.
- Checkpoint'ler Modal Volume'da kalır (`/data/runs/tr-l4`).

İzleme:

```bash
modal run modal_train.py::status          # son step, loss, hız
modal app logs <app-id>                    # canlı log
```

## Örnek sentez (checkpoint indirmeden)

```bash
modal run modal_train.py::sample --text "Merhaba dünya, bu bir deneme." --voice MK
modal volume get zerre-tts runs/tr-l4/sample.wav ./sample.wav
```

## Sonucu indir

```bash
modal volume get zerre-tts runs/tr-l4 ./runs/tr-l4
```

## GPU seçimi ve maliyet

| GPU | Yaklaşık fiyat | Not |
|---|---|---|
| T4 (16 GB) | ~0.59 $/sa | En ucuz; yavaş, bf16 yok |
| **L4 (24 GB)** | **~0.80 $/sa** | Bu model için dengeli seçim |
| A10G (24 GB) | ~1.10 $/sa | L4'ten hızlı |
| A100 40 GB | ~2.10 $/sa | Gerekmiyor, model 4.5M parametre |

11 saat L4 ≈ **9 $**. Modal'ın aylık ücretsiz kredisi (≈30 $) bunu karşılar.

## Bilinen notlar

- `annotations.json` içindeki `acoustic_root` Windows yolu taşıyor; `zerre.data`
  bunu şu sırayla çözer: açık argüman → `ZERRE_ACOUSTIC_ROOT` → kayıtlı yol →
  `annotations/../features_full`. Modal'da dizin yapısı ` /data/annotations` +
  `/data/features_full` olduğu için ek ayar gerekmez.
- Fonemleme eğitim sırasında gerekmez (token'lar hazır), yani konteynerde eSpeak
  şart değil; yine de `sample()` için imaja kuruldu.
- `--max-seconds` dolduğunda çıkması normaldir; tekrar `modal run` ile devam eder.
