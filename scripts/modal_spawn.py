"""Spawn a durable training run on the already-deployed Modal app.

`modal run` keeps an ephemeral app alive only while the local client waits. This
script instead calls a function of a *deployed* app, so the run survives the
laptop shutting down, losing the network or sleeping. Nothing here is started
automatically: run it yourself when you want that guarantee.

Setup once (deploys the app, does not start any work):

    modal deploy modal_train.py

Then, at any time:

    uv run python scripts/modal_spawn.py --hours 6 --name tr-v1
    uv run python scripts/modal_spawn.py --hours 6 --name tr-v1 --status

Watch it later (no local process needed):

    modal app logs zerre-tts-train
"""

import argparse
import sys

APP_NAME = "zerre-tts-train"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="tr-v1")
    parser.add_argument("--steps", type=int, default=65000)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--batch-frames", type=int, default=9000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--gan-start", type=int, default=4000)
    parser.add_argument("--dry-run", action="store_true", help="only print what would be spawned")
    args = parser.parse_args(argv)

    payload = {
        "steps": args.steps,
        "hours": args.hours,
        "name": args.name,
        "batch_frames": args.batch_frames,
        "max_batch": args.max_batch,
        "gan_start": args.gan_start,
    }
    if args.dry_run:
        print(f"would spawn {APP_NAME}.train with {payload}")
        return 0

    import modal

    try:
        train = modal.Function.from_name(APP_NAME, "train")
    except Exception as error:  # noqa: BLE001 - the usual cause is "not deployed yet"
        raise SystemExit(
            f"{APP_NAME} is not deployed (or has no 'train' function): {error}\n"
            "run: modal deploy modal_train.py"
        ) from None

    call = train.spawn(**payload)
    print(f"spawned {APP_NAME}.train\ncall id: {call.object_id}")
    print("this run continues on Modal even if this machine shuts down.\n"
          "watch it with: modal app logs " + APP_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
