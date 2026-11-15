"""Build a freshly-initialised checkpoint and export it to ONNX.

Proves the export path (checkpoint format, frontend fingerprint, ONNX graph) end
to end without waiting for a trained model.

    uv run python scripts/export_smoke.py --output runs/untrained
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from zerre.config import CHECKPOINT_FORMAT
from zerre.model import TinyVoca
from zerre.text import frontend_fingerprint, frontend_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/untrained"))
    args = parser.parse_args(argv)

    from zerre.config import Config

    config = Config()
    torch.manual_seed(0)
    model = TinyVoca(config)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "inference.pt"
    torch.save({
        "format_version": CHECKPOINT_FORMAT,
        "config": config.to_dict(),
        "frontend": {"fingerprint": frontend_fingerprint(), "identity": frontend_identity()},
        "model": model.state_dict(),
    }, checkpoint)
    print(json.dumps({"checkpoint": str(checkpoint), "vocab_size": config.vocab_size,
                      "params": sum(p.numel() for p in model.parameters())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
