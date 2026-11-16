"""Quantize the exported ONNX model to int8 and report the size change.

    uv run python scripts/quantize_onnx.py \
        --input runs/tr-v1/model.onnx --output runs/tr-v1/model-int8.onnx
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/tr-v1/model.onnx"))
    parser.add_argument("--output", type=Path, default=Path("runs/tr-v1/model-int8.onnx"))
    args = parser.parse_args(argv)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    before = args.input.stat().st_size
    quantize_dynamic(str(args.input), str(args.output), weight_type=QuantType.QInt8)
    after = args.output.stat().st_size
    print(f"{args.input.name}: {before / 1e6:.2f} MB")
    print(f"{args.output.name}: {after / 1e6:.2f} MB  ({after / before * 100:.1f}% of the original)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
