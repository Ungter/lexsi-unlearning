"""Block until one of several Vast runs finishes (STATUS DONE/FAILED) or its run.log matches a pattern.

  python scripts/vast/watch.py RUN_ID [RUN_ID ...] [--pattern REGEX] [--all] [--timeout-min 240] [--every 60]

Meant to run as a background job so its exit wakes the caller. Prints, per run, STATUS and the last log lines that
match '== |[done]|FAILED|Traceback|step' when it returns. Exit code 0 = condition met, 2 = timeout.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "vast"), str(ROOT / "src")]

import launch  # noqa: E402
from lfm3u import storage  # noqa: E402

KEEP = re.compile(r"^== |\[done\]|FAILED|Traceback|Error|\"step\"|\[dev\]|\[selfgen\]|\[ft ")


def snapshot(run_id: str) -> tuple[str, str]:
    st = launch.load_state(run_id)
    status = (storage.get_text(f"{st['s3_prefix']}/STATUS") or "").strip()
    log = storage.get_text(f"{st['s3_prefix']}/run.log") or ""
    return status, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--pattern", default=None, help="also return when a run.log matches this regex")
    ap.add_argument("--all", action="store_true", help="wait for every run instead of the first")
    ap.add_argument("--timeout-min", type=float, default=240)
    ap.add_argument("--every", type=float, default=60)
    a = ap.parse_args()
    launch.load_secrets()
    pat = re.compile(a.pattern) if a.pattern else None
    t0, done = time.time(), set()
    while True:
        hit = []
        for r in a.run_ids:
            try:
                status, log = snapshot(r)
            except Exception as e:  # noqa: BLE001
                status, log = f"(error {e})", ""
            if status.startswith(("DONE", "FAILED")):
                done.add(r)
                hit.append(r)
            elif pat and pat.search(log):
                hit.append(r)
        if (a.all and len(done) == len(a.run_ids)) or (not a.all and hit):
            break
        if time.time() - t0 > a.timeout_min * 60:
            print(f"timeout after {a.timeout_min} min")
            for r in a.run_ids:
                print(r, snapshot(r)[0])
            sys.exit(2)
        time.sleep(a.every)
    for r in a.run_ids:
        status, log = snapshot(r)
        lines = [ln for ln in log.splitlines() if KEEP.search(ln)][-12:]
        print(f"### {r}: {status or 'RUNNING'}\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
