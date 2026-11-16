"""Print the headline numbers from an evaluation report."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path("eval/tr-v1/report.json"))
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(f"checkpoint {report.get('checkpoint')}  step {report.get('step')}  asr {report.get('asr_model')}")
    for name, block in (report.get("summary") or {}).items():
        wer = block.get("wer")
        generated = block.get("synthesised_wer", wer)
        reference = block.get("recorded_wer")
        line = f"{name:12s} clips={block.get('clips', '?'):>3}"
        if generated is not None:
            line += f"  WER(sentez)={generated:.3f}"
        if reference is not None:
            line += f"  WER(kayıt)={reference:.3f}"
        for key in ("duration_ratio", "mel_dtw", "generated_pitch_std_semitones"):
            if key in block:
                line += f"  {key}={block[key]}"
        print(line)
    print("frontend token mismatches:", report.get("frontend_token_mismatches"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
