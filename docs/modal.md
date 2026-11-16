# Modal'da eğitim

Modal sunucusuz GPU kiralar; 4090 aramana gerek yok. Hazır dosya: `modal_train.py`.

## Hız ölçümü (bu makine)

| Yön | Hız |
|---|---:|
| İndirme | 68.5 Mbps |
| **Yükleme** | **14.2 Mbps** |

Buna göre iki yol var:

| Yol | Yüklenecek | Süre | Modal'da ek iş |
|---|---:|---:|---:|
| **A. Sadece parquet** (önerilen) | 1.38 GB | **~13 dk** | prepare + hizalama, ~30 dk |
| B. Hazır paket | 6.9 GB | ~65 dk | yok |

## Yol A — parquet yükle, gerisini Modal'da hesapla

```bash
uv pip install modal
modal token new                      # tarayıcı, senin hesabın
modal volume create zerre-tts

# 1.38 GB
modal volume put zerre-tts data/tr/full.parquet /raw/full.parquet

# özellikler + hizalama (L4'te ~30 dk; CPU'da 1.5 saat olan hizalama GPU'da dakikalar)
modal run modal_train.py::data_stage

# eğitim
modal run modal_train.py --steps 65000 --hours 11
```

`data_stage` şunları yapar: `prepare_parallel` (14 işçi) → validation/test
özellikleri → `zerre.annotate --device cuda` → `package_training` bütünlük
kontrolü. Hepsi `/data` volume'una yazar.

## Yol B — hazır paketi yükle

```bash
modal volume put zerre-tts data/tr/annotations /annotations
modal volume put zerre-tts data/tr/features_full /features
modal run modal_train.py --steps 65000 --hours 11
```

## İzleme ve örnek

```bash
modal run modal_train.py::status
modal app logs <app-id>

modal run modal_train.py::sample --text "Merhaba dünya, bu bir deneme." --voice MK
modal volume get zerre-tts runs/tr-l4/sample.wav ./sample.wav
modal volume get zerre-tts runs/tr-l4 ./runs/tr-l4
```

## GPU ve maliyet

| GPU | Yaklaşık | Not |
|---|---|---|
| T4 16 GB | ~0.59 $/sa | En ucuz, yavaş |
| **L4 24 GB** | **~0.80 $/sa** | Bu model için dengeli |
| A10G 24 GB | ~1.10 $/sa | L4'ten hızlı |
| A100 40 GB | ~2.10 $/sa | Gerekmiyor |

11 saat L4 ≈ **9 $** + 30 dk veri işi ≈ 0.40 $. Modal'ın aylık ücretsiz kredisi
(~30 $) bunu karşılar.

## PC kapansa da sürsün (kalıcı app)

`modal run` ile başlatılan koşu, yerel istemci bağlantısı koptuğunda Modal
tarafından durdurulabilir (en fazla son checkpoint'e kadar kayıp, ≤1000 step).
Garantili sürdürmek için app'i bir kez deploy et, sonra işi **spawn** et:

```bash
modal deploy modal_train.py                      # app'i kurar, iş başlatmaz
uv run python scripts/modal_spawn.py --hours 6 --name tr-v1
```

Spawn edilen koşu Modal'da bağımsız çalışır: laptop kapansa, uykuya girse,
internet gitse de devam eder. İzlemek için yerel süreç gerekmez:

```bash
modal app logs zerre-tts-train
uv run python scripts/modal_spawn.py --dry-run   # ne göndereceğini göster
```

Checkpoint mantığı aynı: `last.pt` varsa `--resume` ile kaldığı yerden başlar,
yani spawn'ı tekrar çağırmak koşuyu sürdürür.

### Çalışan işi durdurmadan devralma

`modal run` ile koşan işi, kalıcı app'e **iş kaybetmeden** taşımak için: deploy
edilmiş app'te `--takeover` ile eski app id'sini ver.

```bash
modal deploy modal_train.py

# önce ne olacağını gör (hiçbir şey durdurmaz, hiçbir şey başlatmaz)
uv run python scripts/modal_spawn.py --takeover ap-XXXXXXXX --dry-run

# gerçek devralma: eski app'i durdurur, 15 sn bekler, sonra son checkpoint'ten devam eder
uv run python scripts/modal_spawn.py --takeover ap-XXXXXXXX --hours 6 --name tr-v1
```

Neden kayıp yok: yeni koşu `last.pt`'den `--resume` yapar. Yalnızca son
checkpoint'ten sonraki adımlar (en fazla 1000 step ≈ 8 dakika compute ≈ 2 sent)
tekrar edilir; işlenmiş veri, hizalama ve checkpoint'ler yerinde kalır.

Eski app'i durdurmak zorunlu: iki konteyner aynı run dizinine yazarsa checkpoint
dosyaları çakışır. `--takeover` bunu sırayla yapar (önce durdur, sonra başlat).

## Para yakmadan ilerleme (önerilen sıra)

Her adım tek tek ölçülür, gereksiz GPU dakikası harcanmaz:

| Adım | Komut | Süre | Yaklaşık |
|---|---|---|---:|
| Veri işi | `modal volume put ...` + `modal run modal_train.py::data_stage` | ~30 dk | 0.40 $ |
| CUDA smoke | `modal run modal_train.py --steps 20 --hours 0.1 --name smoke` | ~2 dk | 0.03 $ |
| İlk koşu | `modal run modal_train.py --hours 4 --steps 65000` | 4 sa | 3.20 $ |
| Devam | aynı komut (resume) | 4 sa | 3.20 $ |

Toplam 11 saatlik eğitim ≈ **6.5-8 $**. Konteyner boşta para yakmaz: fonksiyon
biter, GPU kapanır. `--max-seconds` sınırı her koşuyu kapatır.

## Teknik notlar

- `annotations.json` içindeki `acoustic_root` Windows yolu taşır; `zerre.data`
  sırayla çözer: açık argüman → `ZERRE_ACOUSTIC_ROOT` → kayıtlı yol →
  `annotations/../features_full`. Modal'da ek ayar gerekmez.
- Eğitim sırasında fonemlemeye gerek yok (token'lar hazır); eSpeak yalnızca
  `data_stage` ve `sample` için imajda.
- Hizalayıcı (`wav2vec2-lv-60-espeak-cv-ft`) imaja gömülür, çünkü kod onu
  `local_files_only=True` ile yükler.
- `--max-seconds` dolunca çıkar; aynı `modal run` komutu `last.pt`'den devam eder.
- Sürümler yerelde üretilen özelliklerle uyumlu olsun diye sabitlendi
  (numpy, librosa, soundfile, soxr, pyarrow).
