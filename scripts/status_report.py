"""Render the project dashboard from the real state on disk.

Nothing is hardcoded: the numbers come from the parquet, the feature directory,
the annotation journal, the lexicon and runs/. Re-run it whenever you want an
up-to-date view; the page also refreshes itself every 30 seconds.

    uv run python scripts/status_report.py --open
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tr"
FEATURES = DATA / "features_full"
ANNOTATIONS = DATA / "annotations"
RUNS = ROOT / "runs"
WEB = ROOT / "web" / "public"

STAGES = [
    ("frontend", "Türkçe frontend", "envanter + normalizasyon"),
    ("data", "Korpus", "HF veri setleri → 24 kHz parquet"),
    ("features", "Akustik özellikler", "mel + F0 + ötümlülük + enerji"),
    ("annotate", "Hizalama", "CTC ile fonem süreleri"),
    ("train", "Eğitim", "4.5M parametre, kiralık GPU"),
    ("export", "Export", "ONNX + model.json"),
    ("web", "Demo", "tarayıcıda sentez"),
]


def parquet_stats():
    path = DATA / "full.parquet"
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["duration_seconds", "voice_id"])
    durations = table.column("duration_seconds").to_pylist()
    voices = table.column("voice_id").to_pylist()
    by_voice = {}
    for voice, duration in zip(voices, durations):
        by_voice[voice] = by_voice.get(voice, 0.0) + float(duration or 0)
    return {"clips": len(durations), "hours": sum(by_voice.values()) / 3600,
            "by_voice": {voice: hours / 3600 for voice, hours in by_voice.items()},
            "bytes": path.stat().st_size}


def line_count(path):
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def feature_stats(total_clips):
    done = 0
    for split in ("train", "validation", "test"):
        directory = FEATURES / split
        if directory.exists():
            done += sum(1 for _ in directory.glob("*.pt"))
    return {"clips": done, "expected": total_clips}


def annotation_stats(total_clips):
    raw = ANNOTATIONS / "raw" / "train.jsonl"
    aligned = line_count(raw)
    accepted = line_count(ANNOTATIONS / "train.jsonl")
    rejected = line_count(ANNOTATIONS / "train.rejected.jsonl")
    metadata = ANNOTATIONS / "annotations.json"
    complete = False
    summary = None
    if metadata.exists():
        try:
            summary = json.loads(metadata.read_text(encoding="utf-8"))
            complete = bool(summary.get("complete"))
        except json.JSONDecodeError:
            pass
    return {"aligned": aligned, "expected": total_clips, "accepted": accepted,
            "rejected": rejected, "complete": complete}


def lexicon_stats():
    path = WEB / "lexicon-tr.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"words": len(payload.get("words", {})), "bytes": path.stat().st_size}


def run_stats():
    journals = sorted(RUNS.glob("*/journal.jsonl")) if RUNS.exists() else []
    trained = [path.parent / "inference.pt" for path in journals]
    checkpoints = [path for path in trained if path.exists()]
    steps = 0
    if journals:
        last = [line for line in journals[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
        if last:
            try:
                steps = json.loads(last[-1]).get("step", 0)
            except json.JSONDecodeError:
                steps = 0
    return {"checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints], "steps": steps,
            "smoke_only": [str(path.relative_to(ROOT)) for path in RUNS.glob("*/inference.pt") if not path.exists()]}


def web_stats():
    model = WEB / "model.onnx"
    return {"model_bytes": model.stat().st_size if model.exists() else 0}


def git_state():
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("git", "log", "-1", "--oneline"), "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD")}


def percent(value, total):
    return 0.0 if not total else max(0.0, min(1.0, value / total))


def build(state):
    counts = {}
    counts["frontend"] = 1.0
    counts["data"] = 1.0 if state["parquet"] else 0.0
    counts["features"] = percent(state["features"]["clips"], state["features"]["expected"] or 1)
    counts["annotate"] = (1.0 if state["annotation"]["complete"] else
                          percent(state["annotation"]["aligned"], state["annotation"]["expected"] or 1))
    counts["train"] = 1.0 if state["runs"]["checkpoints"] else 0.0
    counts["export"] = 1.0 if state["web"]["model_bytes"] else 0.0
    counts["web"] = 1.0 if state["web"]["model_bytes"] else 0.0
    return counts


def render(state, counts):
    generated = dt.datetime.now().strftime("%d.%m.%Y %H:%M")
    rows = []
    for key, title, subtitle in STAGES:
        value = counts.get(key, 0.0)
        done = "tamam" if value >= 0.999 else (f"%{value * 100:.0f}" if value > 0 else "bekliyor")
        rows.append(f"""
        <tr>
          <td class="name"><strong>{title}</strong><span>{subtitle}</span></td>
          <td class="bar"><div class="track"><div class="fill {'ok' if value >= 0.999 else ''}" style="width:{value * 100:.1f}%"></div></div></td>
          <td class="state {'done' if value >= 0.999 else ''}">{done}</td>
        </tr>""")

    parquet = state["parquet"] or {}
    voices = ", ".join(f"{voice} {hours:.1f} sa" for voice, hours in sorted((parquet.get("by_voice") or {}).items()))
    annotation = state["annotation"]
    cards = [
        ("Korpus", f"{parquet.get('clips', 0):,} klip", voices or "—"),
        ("Özellikler", f"{state['features']['clips']:,} / {state['features']['expected']:,}",
         "mel + prosodi dosyası"),
        ("Annotasyon", f"{annotation['aligned']:,} / {annotation['expected']:,}",
         f"kabul {annotation['accepted']:,} · red {annotation['rejected']:,}"),
        ("Sözlük", f"{state['lexicon']['words']:,} kelime" if state["lexicon"] else "—",
         f"{(state['lexicon'] or {}).get('bytes', 0) / 1e6:.2f} MB"),
        ("Eğitim", f"{state['runs']['steps']:,} step" if state["runs"]["steps"] else "başlamadı",
         f"{len(state['runs']['checkpoints'])} checkpoint"),
        ("Model", f"{state['web']['model_bytes'] / 1e6:.1f} MB" if state["web"]["model_bytes"] else "yok",
         "web/public/model.onnx"),
    ]
    cards_html = "".join(f"""
      <div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>"""
                        for label, value, hint in cards)

    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>zerre-tts — durum</title>
<style>
:root {{ color-scheme: light dark; --fg:#12141a; --bg:#f7f7f5; --muted:#68707d; --line:#dfe2e7;
        --accent:#0b6b5b; --card:#fff; --ok:#0b6b5b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#eceef1; --bg:#0f1113; --muted:#98a2b3;
        --line:#262b31; --accent:#5ad2b8; --card:#161a1e; --ok:#5ad2b8; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2.5rem 1rem 4rem; background:var(--bg); color:var(--fg);
        font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:56rem; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .2rem; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); margin:0 0 2rem; font-variant-numeric:tabular-nums; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:.75rem; margin-bottom:2rem; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.85rem 1rem; }}
.label {{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }}
.value {{ font-size:1.25rem; font-weight:650; margin:.15rem 0 .1rem; font-variant-numeric:tabular-nums; }}
.hint {{ color:var(--muted); font-size:.8rem; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
td {{ padding:.8rem 1rem; border-bottom:1px solid var(--line); vertical-align:middle; }}
tr:last-child td {{ border-bottom:0; }}
.name strong {{ display:block; font-weight:600; }}
.name span {{ color:var(--muted); font-size:.8rem; }}
.bar {{ width:42%; }}
.track {{ height:6px; background:var(--line); border-radius:3px; overflow:hidden; }}
.fill {{ height:100%; background:var(--accent); opacity:.55; }}
.fill.ok {{ opacity:1; }}
.state {{ width:6rem; text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }}
.state.done {{ color:var(--ok); font-weight:600; }}
footer {{ color:var(--muted); font-size:.8rem; margin-top:1.5rem; }}
code {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:.1rem .35rem; }}
</style></head>
<body><main>
<h1>zerre-tts · ilerleme</h1>
<p class="sub">{generated} itibarıyla · <code>{state['git']['commit'] or 'commit yok'}</code> · sayfa 30 sn'de bir yenilenir</p>
<div class="cards">{cards_html}</div>
<table>{''.join(rows)}</table>
<footer>Yenilemek için: <code>uv run python scripts/status_report.py</code> ·
veri ve ağırlıklar repoda tutulmaz · https://github.com/erkinemreta1/zerre-tts</footer>
</main></body></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "status.html")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    parquet = parquet_stats()
    state = {
        "parquet": parquet,
        "features": feature_stats(parquet["clips"] if parquet else 0),
        "annotation": annotation_stats(parquet["clips"] if parquet else 0),
        "lexicon": lexicon_stats(),
        "runs": run_stats(),
        "web": web_stats(),
        "git": git_state(),
    }
    counts = build(state)
    html = render(state, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    summary = {key: round(value, 3) for key, value in counts.items()}
    print(json.dumps({"output": str(args.output), "stages": summary,
                      "features": state["features"], "annotation": state["annotation"]}, indent=2))
    if args.open:
        import webbrowser
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
