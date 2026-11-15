"""Compare F0 extractors on one clip: speed and agreement."""

import io
import time

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

table = pq.ParquetFile("data/tr/full.parquet").read_row_group(0)
waves = []
for row in table.slice(0, 3).to_pylist():
    wave, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    waves.append((wave, rate))
print("clips:", [(len(w) / r, r) for w, r in waves])

SR, HOP, NFFT = 24000, 256, 1024


def timeit(label, fn):
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    audio_seconds = sum(len(w) / r for w, r in waves)
    print(f"{label:28s} {elapsed:6.2f}s  ({audio_seconds/elapsed:5.1f}x realtime)")
    return out


def pyin_default():
    return [librosa.pyin(w, fmin=50, fmax=600, sr=SR, frame_length=NFFT, hop_length=HOP, center=True, pad_mode="constant")[0] for w, _ in waves]


def pyin_cheap():
    return [librosa.pyin(w, fmin=50, fmax=600, sr=SR, frame_length=NFFT, hop_length=HOP, center=True, pad_mode="constant", n_thresholds=30)[0] for w, _ in waves]


def yin_fast():
    return [librosa.yin(w, fmin=50, fmax=600, sr=SR, frame_length=NFFT, hop_length=HOP, center=True, pad_mode="constant") for w, _ in waves]


def torchaudio_pitch():
    import torch
    import torchaudio

    values = []
    for wave, _ in waves:
        tensor = torch.from_numpy(wave)[None]
        freq = torchaudio.functional.detect_pitch_frequency(tensor, SR, frame_time=HOP / SR)
        values.append(freq[0].numpy())
    return values


reference = timeit("librosa.pyin (default)", pyin_default)
cheap = timeit("librosa.pyin (30 thresholds)", pyin_cheap)
yin = timeit("librosa.yin", yin_fast)
ta = timeit("torchaudio detect_pitch", torchaudio_pitch)

for index, (ref, fast) in enumerate(zip(reference, yin)):
    mask = np.isfinite(ref) & np.isfinite(fast)
    if mask.any():
        print(f"clip {index}: yin vs pyin median abs diff {np.median(np.abs(ref[mask]-fast[mask])):.1f} Hz, "
              f"voiced agreement {np.mean(np.isfinite(ref)==np.isfinite(fast)):.2f}")
for index, (ref, fast) in enumerate(zip(reference, ta)):
    freq = np.where(fast > 0, fast, np.nan)
    mask = np.isfinite(ref) & np.isfinite(freq)
    if mask.any():
        print(f"clip {index}: torchaudio vs pyin median abs diff {np.median(np.abs(ref[mask]-freq[mask])):.1f} Hz, "
              f"voiced agreement {np.mean(np.isfinite(ref)==np.isfinite(freq)):.2f}")
