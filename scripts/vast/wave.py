"""Launch many Vast jobs in parallel: one `launch.py up` argument line per line of a wave file.

  python scripts/vast/wave.py WAVE_FILE [--workers 6]

Lines starting with # are skipped; ${T} and ${R} in a line expand to $TARGET_URI / $RETAIN_URI from the environment.
Parallel launches can race for the same offer; launch.py re-picks on no_such_ask. Prints one line per launch with the
run id (or the error) and writes results/vast/waves/<wave>-<time>.json.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(line: str) -> dict:
    args = shlex.split(os.path.expandvars(line))
    env = os.environ | {"PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, str(ROOT / "scripts/vast/launch.py"), "up", *args], capture_output=True, text=True,
                       cwd=ROOT, env=env)
    out = p.stdout + p.stderr
    rid = re.search(r'"run_id": "([^"]+)"', out)
    ok = p.returncode == 0 and "created" in out
    gpu = re.findall(r'"gpu": "([^"]+)"', out)
    return {"line": line, "run_id": rid.group(1) if rid else None, "ok": ok, "gpu": gpu[-1] if gpu else None,
            "tail": out.strip().splitlines()[-1][-300:] if out.strip() else ""}


def main() -> None:
    wave = Path(sys.argv[1])
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    lines = [ln.strip() for ln in wave.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    res = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {}
        for i, ln in enumerate(lines):
            futs[ex.submit(run, ln)] = ln
            time.sleep(2)  # stagger so parallel picks see each other's rentals
        for f in cf.as_completed(futs):
            r = f.result()
            res.append(r)
            print(f"{'OK ' if r['ok'] else 'ERR'} {r['run_id']} {r['gpu']} | {r['line'][:110]}" + ("" if r["ok"] else f"\n    {r['tail']}"),
                  flush=True)
    out = ROOT / "results/vast/waves" / f"{wave.stem}-{time.strftime('%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{sum(r['ok'] for r in res)}/{len(res)} launched; {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
