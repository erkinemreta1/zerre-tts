"""Train zerre-tts on Modal (serverless GPU).

Setup once:
    pip install modal
    modal token new                 # opens the browser, needs your account
    modal volume create zerre-tts   # persistent store for data and checkpoints

Upload the prepared bundle once (6.9 GB):
    modal volume put zerre-tts data/tr/annotations /annotations
    modal volume put zerre-tts data/tr/features_full /features

Train (L4 is the cheapest GPU that fits this model):
    modal run modal_train.py --steps 65000 --hours 11

Synthesize from the latest checkpoint without downloading it:
    modal run modal_train.py::sample --text "Merhaba dünya, bu bir deneme."

Pull the result back:
    modal volume get zerre-tts runs/tr-l4 ./runs/tr-l4
"""

import modal

APP_NAME = "zerre-tts-train"
VOLUME = "zerre-tts"
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "espeak-ng", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "numpy<3", "soundfile", "filelock", "librosa>=0.10.2",
        "soxr", "pyarrow",
    )
    .add_local_dir("src/zerre", remote_path="/root/zerre/zerre")
    .env({"PYTHONPATH": "/root/zerre", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8"})
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(gpu="L4", timeout=12 * 3600, volumes={DATA: volume}, cpu=8, memory=16384)
def train(steps: int = 65000, hours: float = 11.0, name: str = "tr-l4",
          batch_frames: int = 9000, max_batch: int = 32, gan_start: int = 4000):
    import subprocess
    from pathlib import Path

    output = f"{DATA}/runs/{name}"
    command = [
        "python", "-m", "zerre.train",
        "--train", f"{DATA}/annotations/train.jsonl",
        "--validation", f"{DATA}/annotations/validation.jsonl",
        "--output", output,
        "--device", "cuda",
        "--steps", str(steps),
        "--batch-frames", str(batch_frames),
        "--max-batch", str(max_batch),
        "--gan-start", str(gan_start),
        "--save-every", "1000",
        "--keep-every", "10000",
        "--log-every", "50",
        "--max-seconds", str(int(hours * 3600)),
    ]
    latest = Path(output) / "last.pt"
    if latest.exists():
        command += ["--resume", str(latest)]
        print(f"resuming from {latest}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    volume.commit()
    return output


@app.function(gpu="L4", timeout=1800, volumes={DATA: volume})
def sample(text: str, voice: str = "MK", name: str = "tr-l4", step: int | None = None):
    """Synthesize one sentence from a checkpoint on the volume and save a wav."""
    import subprocess

    checkpoint = (
        f"{DATA}/runs/{name}/step-{step:06d}.pt" if step else f"{DATA}/runs/{name}/inference.pt"
    )
    output = f"{DATA}/runs/{name}/sample.wav"
    subprocess.run(
        ["python", "-m", "zerre.infer", "--checkpoint", checkpoint, "--lang", "tr",
         "--text", text, "--voice", voice, "--output", output],
        check=True,
    )
    volume.commit()
    return output


@app.function(timeout=600, volumes={DATA: volume})
def status(name: str = "tr-l4"):
    from pathlib import Path

    directory = Path(f"{DATA}/runs/{name}")
    if not directory.exists():
        return {"exists": False}
    files = sorted(path.name for path in directory.iterdir())
    journal = directory / "metrics.jsonl"
    last = {}
    if journal.exists():
        lines = [line for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            import json
            last = json.loads(lines[-1])
    return {"exists": True, "files": files, "last": {key: last.get(key) for key in
            ("step", "loss", "duration_length_error", "voicing_accuracy", "steps_per_second")}}


@app.local_entrypoint()
def main(steps: int = 65000, hours: float = 11.0, name: str = "tr-l4"):
    print(train.remote(steps=steps, hours=hours, name=name))
