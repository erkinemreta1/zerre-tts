# Yayın metinleri

Hazır kopyala-yapıştır. Hepsi aynı çekirdeği söyler: **8.5 MB, tarayıcıda, Türkçe,
Apache-2.0, sıfırdan eğitim.**

Bağlantılar:
- Demo: https://erkinemreta1.github.io/zerre-tts/
- Ağırlıklar: https://huggingface.co/erkint/zerre-tts
- Kod: https://github.com/erkinemreta1/zerre-tts

## X / Twitter (TR)

> 8.5 MB'lık bir Türkçe konuşma sentezleyicisi yaptım.
>
> Tarayıcıda çalışıyor, sunucu yok, ses cihazdan çıkmıyor. 4.5M parametre,
> sıfırdan eğitim, Apache-2.0.
>
> Held-out testte WER %1.3, hiç görmediği cümlelerde %12.7.
> Tek L4'te ~7 saatte eğitildi.
>
> Demo: erkinemreta1.github.io/zerre-tts
> Ağırlıklar: huggingface.co/erkint/zerre-tts
>
> Sınırlar: iki ses, robotik ton, klonlama yok.

## X / Twitter (EN)

> I trained a 4.5M-parameter Turkish TTS from scratch. 8.5 MB.
>
> It runs in a browser tab — no server, no API, the audio never leaves the device.
> Apache-2.0.
>
> WER 1.3% on held-out clips, 12.7% on sentences it has never seen.
> ~7 h on one L4.
>
> Demo: erkinemreta1.github.io/zerre-tts
> Weights: huggingface.co/erkint/zerre-tts

## Reddit · r/LocalLLaMA

**Başlık:** [P] 8.5 MB Turkish TTS trained from scratch, runs in the browser (Apache-2.0)

> I wanted to know how little a usable TTS needs, so I ported tiny-voca's
> architecture (4.5M params, source-filter vocoder, no neural vocoder) to Turkish
> and trained it from scratch.
>
> - 8.5 MB int8 ONNX, 21.4 MB fp32. WebAssembly, ~200 ms for 3.4 s of audio.
> - Two voices, no cloning, no streaming.
> - Data: 12 h of openly licensed Turkish audiobook recordings, aligned with an
>   eSpeak phoneme recogniser; 65k steps, ~7 h on one L4 (~$7).
> - WER (Whisper large-v3, Turkish): 1.3% on 20 held-out clips, 12.7% on 8 new
>   sentences. Small sets — treat as a sanity check, not a benchmark.
>
> Honest limits: intonation is still flat and slightly robotic; questions and
> exclamations are weak; foreign names are often wrong. Quantisation to int8
> changed nothing measurable, which surprised me.
>
> Demo / repo / weights in the comments.
>
> Happy to answer anything about the alignment pipeline or the export.

İlk yorum:

> Links:
> - Demo: https://erkinemreta1.github.io/zerre-tts/
> - Code + training recipe: https://github.com/erkinemreta1/zerre-tts
> - Weights: https://huggingface.co/erkint/zerre-tts
>
> What made this tractable: Turkish orthography is almost phonemic, so eSpeak's
> `tr` voice is a good enough front end, and the language pack is a small module
> — the acoustic model is unchanged from English.

## Hacker News · Show HN

**Başlık:** Show HN: 8.5 MB Turkish text-to-speech that runs in the browser

> A 4.5M-parameter Turkish TTS, trained from scratch, quantised to 8.5 MB and run
> with ONNX Runtime Web. No server: the model downloads once and the audio never
> leaves the tab.
>
> Architecture is FastSpeech-2-style (duration, pitch, voicing, energy predicted
> explicitly) with a source-filter vocoder instead of a neural one — that is what
> keeps it small. Text to phonemes is eSpeak NG `tr`, or in the browser a 25k-word
> lexicon plus rules.
>
> Numbers: WER 1.3% on 20 held-out clips, 12.7% on 8 unseen sentences (Whisper
> large-v3). 65k steps, ~7 h on one L4.
>
> It is not as natural as a 300M+ model and I would not ship the voice as-is; the
> interesting part is the size/quality point. Apache-2.0, code and recipe open.

## LinkedIn (TR)

> Küçük bir Türkçe konuşma sentezleyicisi yayınladım: 8.5 MB, tarayıcıda çalışıyor.
>
> - 4.5M parametre, sıfırdan eğitim, Apache-2.0
> - Sunucu yok; ses tarayıcıdan çıkmıyor
> - Held-out testte WER %1.3, yeni cümlelerde %12.7
> - Tek L4'te ~7 saat, ~7 $
>
> Sınırlarını da yazdım: iki ses, ton hâlâ robotik, klonlama yok. Amaç ürün
> değil, "bu boyutta ne kadar oluyor" sorusuna cevap.
>
> Demo: erkinemreta1.github.io/zerre-tts

## Eleştiri gelirse (hazır cevaplar)

| Eleştiri | Cevap |
|---|---|
| "Doğal değil" | Doğru, 4.5M parametre. Kıyas noktası boyut: XTTS-v2 2 GB. |
| "WER seti çok küçük" | Evet, 20 klip / 8 cümle; kartta da öyle yazıyor. |
| "8.5 MB ama runtime 10 MB daha indiriyor" | Doğru, model dosyası 8.5 MB; ORT WASM ayrı. |
| "Sesler kimin?" | Açık lisanslı kayıtlardan; klonlama yok, kimlik iddiası yok. Ayrıntı `docs/data_sources.md`. |
| "Neden eSpeak?" | Türkçe yazım neredeyse fonemik; kural tabanlı front end yeterli. |

## Paylaşmadan önce kontrol

- [ ] Demo linki mobilde de açılıyor (Safari dahil)
- [ ] HF kartında lisans ve sınırlar doğru
- [ ] README'de demo + ağırlık linkleri var
- [ ] `docs/data_sources.md` erişilebilir
- [ ] Ekran görüntüsü/video: 20-30 sn, "yaz → çal" akışı
