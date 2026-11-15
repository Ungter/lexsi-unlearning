"""One line per Vast run launched since a time: STATUS, last stage, failures logged by the sweep runners.

  python scripts/vast/board.py [--since 20261114-0815] [--grep TEXT]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "vast"), str(ROOT / "src")]

import launch  # noqa: E402
from lfm3u import storage  # noqa: E402

FAILED = ("results/unlearn/failed.txt", "results/audit/failed.txt", "results/failed.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--grep", default="")
    a = ap.parse_args()
    launch.load_secrets()
    for f in sorted((ROOT / "results/vast").glob("*.json")):
        rid = f.stem
        m = re.search(r"(\d{8}-\d{6})", rid)
        if not m or m.group(1) < a.since or (a.grep and a.grep not in rid):
            continue
        st = json.loads(f.read_text())
        p = st["s3_prefix"]
        status = (storage.get_text(f"{p}/STATUS") or "-").strip()[:40]
        log = storage.get_text(f"{p}/run.log") or ""
        stages = [ln for ln in log.splitlines() if ln.startswith("== ")]
        fails = sum(len((storage.get_text(f"{p}/{x}") or "").strip().splitlines()) for x in FAILED)
        last = stages[-1][24:110] if stages else ""
        print(f"{rid[-28:]:28} {st.get('gpu', '')[:18]:18} {status:10} fails={fails} | {last}")


if __name__ == "__main__":
    main()
