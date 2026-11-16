"""Compare prosody of the same sentence with and without an exclamation mark."""

import numpy as np
import soundfile as sf

from zerre.infer import load_model, speak

CASES = {
    "nokta": "Merhaba dünya, bugün hava çok güzel.",
    "unlem": "Merhaba dünya, bugün hava çok güzel!",
    "cok_unlem": "Merhaba dünya, bugün hava çok güzel!!!",
}

model = load_model("runs/tr-v1/inference.pt", "cpu", check_frontend=False)


def f0_stats(wave, rate=24000):
    import librosa

    f0 = librosa.yin(wave, fmin=60, fmax=800, sr=rate, frame_length=1024, hop_length=256)
    voiced = f0[(f0 > 60) & (f0 < 800)]
    return {
        "f0_mean": round(float(np.mean(voiced)), 1) if voiced.size else None,
        "f0_max": round(float(np.max(voiced)), 1) if voiced.size else None,
        "f0_std": round(float(np.std(voiced)), 1) if voiced.size else None,
        "rms": round(float(np.sqrt(np.mean(wave ** 2))), 4),
        "peak": round(float(np.max(np.abs(wave))), 3),
        "seconds": round(len(wave) / rate, 2),
    }


for name, text in CASES.items():
    wave = speak(model, text, "MK")[0]
    data = wave.numpy()
    print(name, f0_stats(data), "| tokens:", " ".join(__import__("zerre.text", fromlist=["phonemize"]).phonemize(text).symbols[-6:]))
    sf.write(f"samples/unlem-{name}.wav", data, 24000, subtype="PCM_16")
