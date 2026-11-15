# Eğitim runbook'u

## Nerede eğitilir

Bu makinede **eğitilmez**: Intel Core Ultra 5 226V + Arc 130V iGPU var, CUDA yok.
Eğitim kiralık bir 4090'da (RunPod / Vast / Lambda) tek gecede koşar; maliyet
~7 $ (upstream 19 saatlik veriyle 7.7 saatte bitirdi).

Yerelde yapılanlar: frontend, veri hazırlama, hizalama, testler ve küçük smoke
koşuları.

## Girdiler

| Dosya | Nereden |
|---|---|
| `data/tr/annotations/train.jsonl` | `zerre.annotate` |
| `data/tr/annotations/validation.jsonl` | `zerre.annotate` (ayrı split) |

Annotation manifesti `path` alanında feature dosyasının yolunu taşır; feature
dizini eğitim makinesinde aynı göreli konumda olmalı.

## Smoke (yerel, CPU)

Amaç: pipeline'ın öğrendiğini görmek, hiperparametre avı değil.

```bash
uv run python -m zerre.train \
  --train data/tr/annotations/train.jsonl \
  --validation data/tr/annotations/validation.jsonl \
  --output runs/smoke --device cpu \
  --subset 64 --steps 60 --batch-frames 3000 --max-batch 8 \
  --gan-start 40 --teacher-hold 20 --teacher-ramp 40 --max-seconds 900
```

Beklenen: `runs/smoke/inference.pt` + `journal.jsonl`. Loss düşüyor mu, NaN var mı,
`--max-seconds` sınırında düzgün kaydediyor mu bakılır.

## Gerçek koşu (4090)

```bash
# makinede
uv venv --python 3.11 && uv pip install -e ".[prepare]"
apt install -y espeak-ng   # annotation zaten yapıldıysa gerekmez
# veri: HF'den data/tr/annotations + data/tr/features_full indirilir

uv run python -m zerre.train \
  --train data/tr/annotations/train.jsonl \
  --validation data/tr/annotations/validation.jsonl \
  --output runs/tr-v1 --device cuda \
  --steps 65000 --batch-frames 9000 --max-batch 32 \
  --gan-start 4000 --gan-ratio 0.7 \
  --save-every 1000 --keep-every 10000 --log-every 25 \
  --max-seconds 21600
```

`--max-seconds` duvar saati sınırı: sınırda checkpoint yazıp çıkar, aynı komutla
devam edilir (`--output` aynı dizin). Spot instance kopmasına karşı bu şart.

Beklenen süre: 8-12 saat (16 saat veri, 65k step). Upstream referansı: 19.1 saat
veri, 104k step, 4090'da 7.7 saat.

## Kapılar

| Step | Ölçüm | Eşik |
|---:|---|---|
| 20.000 | held-out 16 cümle ile WER (Whisper `large-v3-turbo`, Türkçe) | raporla |
| 40.000 | aynı | ≤ %25, yoksa dur |
| 65.000 | aynı + MOS-lite | WER ≤ %10, MOS ≥ 3.5 |

40k'da eşik tutmuyorsa step artırmak yerine veri/hizalamaya dönülür.

## Değerlendirme

```bash
uv run python -m zerre.evaluate --checkpoint runs/tr-v1/inference.pt \
  --annotations data/tr/annotations --output eval/tr-v1
```

Ek olarak planlanan bağımsız setler: Common Voice TR `test` (300 klip) ve FLEURS
`tr` (100 klip); ikisi de eğitime girmez (bkz. plan §4.5).

## Export

```bash
uv run python -m zerre.export --checkpoint runs/tr-v1/inference.pt --output weights/model.onnx
```

ONNX ~18 MB; tarayıcı demosu bunu HF'den çeker.
