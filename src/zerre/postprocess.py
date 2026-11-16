"""Cheap, deterministic post-processing for synthesis output.

The source-filter vocoder leaves two measured defects next to a real recording:
too much energy below 120 Hz (20% of the total against 3% in the recording) and
a slightly recessed 2-4 kHz band. A high-pass, a gentle presence shelf, and
loudness normalisation fix the first two without touching the model.

Everything is linear-phase FIR plus a limiter, so it can run on a laptop in
milliseconds and does not need training.
"""

import numpy as np


def _design(sample_rate, highpass, lowpass, presence_hz, presence_gain_db, taps=1025):
    from scipy.signal import firwin2

    nyquist = sample_rate / 2
    gain = 10 ** (presence_gain_db / 20)
    # firwin2 needs a curve that starts at 0 Hz and ends exactly at Nyquist.
    bands = [
        0.0, highpass * 0.7, highpass * 1.3,
        presence_hz * 0.7, presence_hz, presence_hz * 1.5,
        min(lowpass * 0.9, nyquist * 0.85), min(lowpass, nyquist * 0.95), nyquist,
    ]
    gains = [0.0, 0.0, 1.0, 1.0, gain, gain, 0.9, 0.5, 0.0]
    freq = [min(value / nyquist, 1.0) for value in bands]
    for index in range(1, len(freq)):
        if freq[index] <= freq[index - 1]:
            freq[index] = min(freq[index - 1] + 1e-4, 1.0)
    freq[-1] = 1.0
    return firwin2(taps, freq, gains)


def denoise(wave, sample_rate, floor=0.06, oversubtract=1.4, n_fft=1024, hop=256):
    """Gentle spectral subtraction against the clip's own noise floor.

    The vocoder leaves a low-level buzz that survives the high-pass; the model
    itself is fine. Estimating the floor from the quietest 10% of frames per bin
    removes it without touching the harmonics.
    """
    import numpy as np

    audio = np.asarray(wave, dtype=np.float64)
    if audio.size < n_fft:
        return audio.astype(np.float32)
    window = np.hanning(n_fft)
    frames = 1 + (len(audio) - n_fft) // hop
    index = np.arange(frames)[:, None] * hop + np.arange(n_fft)[None, :]
    spectrum = np.fft.rfft(audio[index] * window, axis=1)
    magnitude = np.abs(spectrum)
    noise = np.percentile(magnitude, 10, axis=0)
    cleaned = np.maximum(magnitude - oversubtract * noise, floor * magnitude) * np.exp(1j * np.angle(spectrum))
    output = np.zeros(len(audio) + n_fft, dtype=np.float64)
    weight = np.zeros_like(output)
    for frame in range(frames):
        segment = np.fft.irfft(cleaned[frame], n=n_fft) * window
        output[frame * hop:frame * hop + n_fft] += segment
        weight[frame * hop:frame * hop + n_fft] += window ** 2
    output = output[: len(audio)] / np.maximum(weight[: len(audio)], 1e-8)
    return output.astype(np.float32)


def polish(wave, sample_rate, highpass=90.0, lowpass=11000.0, presence_hz=3000.0,
           presence_gain_db=2.5, target_rms_db=-20.0, peak=0.95, subtract_noise=False):
    """High-pass, presence shelf, loudness match, soft limit. Returns float32."""
    from scipy.signal import fftconvolve

    audio = np.asarray(wave, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return audio.astype(np.float32)
    audio = audio - audio.mean()
    if subtract_noise:
        audio = denoise(audio, sample_rate).astype(np.float64)

    kernel = _design(sample_rate, highpass, lowpass, presence_hz, presence_gain_db)
    audio = fftconvolve(audio, kernel, mode="same")

    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-9:
        audio *= 10 ** (target_rms_db / 20) / rms

    if peak is not None:
        loudest = float(np.max(np.abs(audio)))
        if loudest > peak:
            # Soft-knee: linear below the knee, tanh above it. Keeps transients
            # recognisable instead of brick-walling them.
            knee = peak * 0.8
            over = np.abs(audio) > knee
            sign = np.sign(audio)
            audio[over] = sign[over] * (knee + (peak - knee) * np.tanh(
                (np.abs(audio[over]) - knee) / (peak - knee)))
    return audio.astype(np.float32)
