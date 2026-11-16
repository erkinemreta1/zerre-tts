"""Train zerre-tts on Modal (serverless GPU).

Two ways to get the data there. The second is five times faster on a normal home
upload (measured here: 14.2 Mbps up, 68.5 Mbps down).

A. Upload the prepared bundle (6.9 GB, ~65 min):
    modal volume put zerre-tts data/tr/annotations /annotations
    modal volume put zerre-tts data/tr/features_full /features

B. Upload only the corpus parquet (1.38 GB, ~13 min) and prepare + align on
   Modal, where the GPU does the alignment in minutes:
    modal volume put zerre-tts data/tr/full.parquet /raw/full.parquet
    modal run modal_train.py::data_stage

Then, in both cases:
    modal run modal_train.py --steps 65000 --hours 11
    modal run modal_train.py::sample --text "Merhaba dünya." --voice MK

Setup once:
    pip install modal
    modal token new
    modal volume create zerre-tts

Pull results back:
    modal volume get zerre-tts runs/tr-l4 ./runs/tr-l4
"""

import modal

APP_NAME = "zerre-tts-train"
VOLUME = "zerre-tts"
DATA = "/data"

def _download_aligner():
    from huggingface_hub import snapshot_download

    snapshot_download("facebook/wav2vec2-lv-60-espeak-cv-ft")


# Versions pinned to the machine the corpus was prepared on, so features and
# tokens stay comparable. torch is left floating: the CUDA build differs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "espeak-ng", "ffmpeg")
    .pip_install(
        "torch>=2.4,<3", "torchaudio>=2.4,<3", "numpy==2.4.6", "soundfile==0.14.0",
        "filelock>=3.13,<4", "librosa==0.11.0", "soxr==1.1.0", "pyarrow==26.0.1",
        "transformers>=4.40,<5", "scipy>=1.10,<2", "huggingface-hub>=0.34,<2",
    )
    .env({"PYTHONPATH": "/root/zerre", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8"})
    # The annotation recogniser is loaded with local_files_only=True, so it has
    # to be in the image rather than fetched at run time.
    .run_function(_download_aligner)
    # add_local_* must be the last build steps: Modal mounts them per container.
    .add_local_dir("src/zerre", remote_path="/root/zerre/zerre")
    .add_local_dir("scripts", remote_path="/root/zerre/scripts")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(gpu="L4", timeout=2 * 3600, volumes={DATA: volume}, cpu=16, memory=32768)
def data_stage(parquet: str = f"{DATA}/raw/full.parquet"):
    """Prepare acoustic features and align the corpus on Modal.

    Input:  the corpus parquet (1.38 GB) at ``parquet``.
    Output: /data/features_full and /data/annotations in the volume.
    """
    import subprocess
    from pathlib import Path

    if not Path(parquet).exists():
        raise SystemExit(f"{parquet} is missing; run: modal volume put {VOLUME} data/tr/full.parquet /raw/full.parquet")

    def run(command):
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)

    run(["python", "scripts/prepare_parallel.py", "--inputs", parquet,
         "--output", f"{DATA}/features_full", "--workers", "14"])
    for split in ("validation", "test"):
        run(["python", "-m", "zerre.prepare", "--inputs", parquet, "--split", split,
             "--output", f"{DATA}/features_full"])
    run(["python", "-m", "zerre.annotate", "--features", f"{DATA}/features_full",
         "--output", f"{DATA}/annotations", "--dataset-revision", "local",
         "--device", "cuda", "--splits", "train", "validation", "test"])
    run(["python", "scripts/package_training.py", "--annotations", f"{DATA}/annotations",
         "--features", f"{DATA}/features_full", "--output", f"{DATA}/reports/training-bundle.json"])
    volume.commit()
    return f"{DATA}/annotations"


@app.function(gpu="L4", timeout=12 * 3600, volumes={DATA: volume}, cpu=8, memory=32768)
def train(steps: int = 65000, hours: float = 11.0, name: str = "tr-l4",
          batch_frames: int = 9000, max_batch: int = 32, gan_start: int = 4000):
    """Train, keeping the GPU fed and reporting how busy it actually was.

    Two things waste money on a rented GPU: reading features from the network
    volume during training, and an under-sized batch that leaves the card idle.
    The features are copied to the container's local disk first (the dataset then
    caches them in RAM), and a monitor thread records utilisation so a run can be
    judged instead of guessed.
    """
    import json
    import os
    import shutil
    import subprocess
    import threading
    import time
    from pathlib import Path

    local_features = "/tmp/features_full"
    if not Path(local_features).exists():
        print(f"copying features to local disk: {DATA}/features_full -> {local_features}", flush=True)
        shutil.copytree(f"{DATA}/features_full", local_features)
    os.environ["ZERRE_ACOUSTIC_ROOT"] = local_features

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

    samples = []
    stop = threading.Event()

    def monitor():
        import torch

        while not stop.is_set():
            if torch.cuda.is_available():
                samples.append(float(torch.cuda.utilization()))
            stop.wait(2.0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    print(" ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        stop.set()
        thread.join(timeout=5)
        if samples:
            report = {"samples": len(samples), "mean_utilization": round(sum(samples) / len(samples), 1),
                      "max_utilization": round(max(samples), 1),
                      "below_50": round(sum(1 for value in samples if value < 50) / len(samples), 3)}
            (Path(output) / "gpu.json").write_text(json.dumps(report), encoding="utf-8")
            print(json.dumps(report), flush=True)
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
    journal = directory / "metrics.jsonl"
    last = {}
    if journal.exists():
        lines = [line for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            import json
            last = json.loads(lines[-1])
    return {"exists": True, "files": sorted(path.name for path in directory.iterdir()),
            "last": {key: last.get(key) for key in
                     ("step", "loss", "duration_length_error", "voicing_accuracy", "steps_per_second")}}


@app.local_entrypoint()
def main(steps: int = 65000, hours: float = 11.0, name: str = "tr-l4"):
    print(train.remote(steps=steps, hours=hours, name=name))
