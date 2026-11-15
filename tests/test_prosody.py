import numpy as np

from zerre.prosody import autocorrelation_strength, voicing

SR = 24000
HOP = 256
NFFT = 1024


def tone(frequency, seconds=1.0, amplitude=0.3):
    time = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_sine_is_voiced_everywhere():
    voiced, strength = voicing(tone(200), SR, HOP, NFFT, 50, 600)
    assert voiced.mean() > 0.95, voiced.mean()
    assert np.median(strength) > 0.8


def test_white_noise_is_unvoiced():
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(SR)).astype(np.float32)
    voiced, _ = voicing(noise, SR, HOP, NFFT, 50, 600)
    assert voiced.mean() < 0.05, voiced.mean()


def test_silence_is_not_voiced():
    voiced, strength = voicing(np.zeros(SR, dtype=np.float32), SR, HOP, NFFT, 50, 600)
    assert not voiced.any()
    assert np.allclose(strength, 0.0)


def test_fricative_after_vowel_breaks_voicing():
    """A loud /s/-like burst must not count as voiced: energy alone would."""
    rng = np.random.default_rng(1)
    vowel = tone(180, 0.5)
    fricative = (0.25 * rng.standard_normal(SR // 2)).astype(np.float32)
    signal = np.concatenate([vowel, fricative])
    voiced, _ = voicing(signal, SR, HOP, NFFT, 50, 600)
    frames = len(voiced)
    first_half, second_half = voiced[: frames // 2 - 4], voiced[frames // 2 + 4:]
    assert first_half.mean() > 0.9
    assert second_half.mean() < 0.05


def test_strength_is_bounded():
    strength = autocorrelation_strength(tone(150), SR, HOP, NFFT, 50, 600)
    assert strength.min() >= 0.0 and strength.max() <= 1.0
