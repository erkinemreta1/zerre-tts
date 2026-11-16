"""Band energy of a wav file, to see where vocoder noise sits."""

import sys

import numpy as np
import soundfile as sf

path = sys.argv[1]
wave, rate = sf.read(path, dtype="float32")
if wave.ndim > 1:
    wave = wave.mean(axis=1)
spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) ** 2
freqs = np.fft.rfftfreq(len(wave), 1 / rate)
total = spectrum.sum()
edges = [0, 60, 120, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000]
print(f"{path}  {len(wave)/rate:.2f}s  {rate} Hz")
for low, high in zip(edges[:-1], edges[1:]):
    mask = (freqs >= low) & (freqs < high)
    share = spectrum[mask].sum() / total
    print(f"{low:>6}-{high:<6} Hz  {share*100:6.2f}%  {'#' * int(share * 120)}")
peak = freqs[int(np.argmax(spectrum))]
print(f"peak bin: {peak:.0f} Hz")
