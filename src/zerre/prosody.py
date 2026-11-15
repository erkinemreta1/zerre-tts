"""Frame-wise voicing from normalised autocorrelation.

The model needs a voicing flag per frame: the synthesiser switches between a
pulse train and noise with it, and the annotation quality checks use it to find
phone boundaries. A pitch tracker that only returns F0 (YIN) says nothing about
voicing, and frame energy is not a substitute: voiceless fricatives are loud.

This module computes the normalised autocorrelation peak in the lag range that
corresponds to the allowed F0 band. Speech frames with a strong periodic peak
are voiced; fricatives and silence are not. It is FFT based, so it costs about
the same as one extra STFT.
"""

import numpy as np


def autocorrelation_strength(wave, sample_rate, hop, frame_length, f_min, f_max):
    """Normalised peak of the autocorrelation per frame, shape ``(frames,)``.

    Frame ``i`` is centred on sample ``i * hop``; the same convention as the
    stored mel features. Values are in ``[0, 1]``.
    """
    wave = np.asarray(wave, dtype=np.float64)
    frames = 1 + (len(wave) - 1) // hop
    padded = np.pad(wave, (frame_length // 2, frame_length // 2 + frame_length))
    n_fft = 1 << int(np.ceil(np.log2(2 * frame_length)))
    starts = np.arange(frames) * hop
    index = starts[:, None] + np.arange(frame_length)[None, :]
    windowed = padded[index]
    windowed = windowed - windowed.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = np.abs(spectrum) ** 2
    correlation = np.fft.irfft(power, n=n_fft, axis=1)[:, :frame_length]
    energy = correlation[:, :1]
    lag_min = max(1, int(np.floor(sample_rate / f_max)))
    lag_max = min(frame_length - 2, int(np.ceil(sample_rate / f_min)))
    if lag_max <= lag_min:
        raise ValueError("empty lag range; check f_min/f_max and frame_length")
    peak = correlation[:, lag_min:lag_max + 1].max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = np.where(energy[:, 0] > 1e-9, peak / energy[:, 0], 0.0)
    return np.clip(strength, 0.0, 1.0)


def voicing(wave, sample_rate, hop, frame_length, f_min, f_max, threshold=0.45, energy_db_below_peak=40.0):
    """Boolean voicing per frame, combining periodicity with an energy gate."""
    strength = autocorrelation_strength(wave, sample_rate, hop, frame_length, f_min, f_max)
    frames = strength.shape[0]
    window = np.hanning(frame_length)
    padded = np.pad(np.asarray(wave, dtype=np.float64), (frame_length // 2, frame_length // 2 + frame_length))
    index = np.arange(frames)[:, None] * hop + np.arange(frame_length)[None, :]
    level = 20 * np.log10(np.maximum(np.sqrt(np.mean((padded[index] * window) ** 2, axis=1)), 1e-8))
    gate = level > np.percentile(level, 95) - energy_db_below_peak
    return (strength >= threshold) & gate, strength
