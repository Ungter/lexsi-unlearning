"""Local client for a Phase 7 worker box (scripts/vast/jobs/surgery_worker.sh): queue a command in S3, wait for it,
print its output. No SSH needed. Commands run with bash in /root/lexsi on the box (use python3 there).

  python scripts/vast/surg_remote.py RUN_ID ready [--wait MIN]          wait for the worker (setup + selftest done)
  python scripts/vast/surg_remote.py RUN_ID run [--put LOCAL[=REMOTE]]... [--timeout MIN] [--tail N] -- CMD...
        --put copies a local file to REMOTE (path relative to /root/lexsi; default: LOCAL's path relative to the repo)
        before CMD runs. Prints the output (last N lines with --tail) and exits with the command's exit code.
        If --timeout passes first, prints the partial output and exits 3; the command keeps running on the box and
        `result ID` picks it up later.
  python scripts/vast/surg_remote.py RUN_ID result ID [--wait MIN]      output of an earlier command
  python scripts/vast/surg_remote.py RUN_ID pull RESULTS_SUBDIR LOCAL_DIR   download $S3_PREFIX/results/RESULTS_SUBDIR
  python scripts/vast/surg_remote.py RUN_ID log [--tail N]              the box's run.log (setup, selftest, worker)
  python scripts/vast/surg_remote.py RUN_ID stop                        end the worker; the box syncs and self-destroys

Example: python scripts/vast/surg_remote.py RUN run --put work/steps/03_probe.py=steps_in/03_probe.py -- \\
             python3 scripts/16_surgery.py exec l2_book_b50 steps_in/03_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch  # noqa: E402  (load_secrets: .env Tigris keys -> AWS_* variables)

launch.load_secrets()
from lfm3u import storage  # noqa: E402


def prefix(run_id: str) -> str:
    st = json.loads((ROOT / "results" / "vast" / f"{run_id}.json").read_text())
    return st["s3_prefix"].rstrip("/")


def print_out(text: str, tail: int | None) -> None:
    lines = text.splitlines()
    if tail and len(lines) > tail:
        print(f"[... {len(lines) - tail} earlier lines; full output: result ID]")
        lines = lines[-tail:]
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", "replace"))
    sys.stdout.flush()


def wait_result(p: str, i: str, minutes: float, tail: int | None) -> int:
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        rc = storage.get_text(f"{p}/done/{i}/rc")
        if rc is not None:
            print_out(storage.get_text(f"{p}/done/{i}/out") or "", tail)
            return int(rc.strip() or 1)
        time.sleep(3)
    partial = storage.get_text(f"{p}/done/{i}/out")
    print_out((partial or "(no output yet)") + f"\n[client] still running after {minutes} min; job id {i}", tail)
    return 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ready")
    r.add_argument("--wait", type=float, default=30)
    r = sub.add_parser("run")
    r.add_argument("--put", action="append", default=[])
    r.add_argument("--timeout", type=float, default=60)
    r.add_argument("--tail", type=int, default=None)
    r.add_argument("command", nargs=argparse.REMAINDER)
    r = sub.add_parser("result")
    r.add_argument("id")
    r.add_argument("--wait", type=float, default=60)
    r.add_argument("--tail", type=int, default=None)
    r = sub.add_parser("pull")
    r.add_argument("subdir")
    r.add_argument("local")
    r = sub.add_parser("log")
    r.add_argument("--tail", type=int, default=60)
    sub.add_parser("stop")
    a = ap.parse_args()
    p = prefix(a.run_id)

    if a.cmd == "ready":
        t0 = time.time()
        while time.time() - t0 < a.wait * 60:
            ready = storage.get_text(f"{p}/READY")
            status = (storage.get_text(f"{p}/STATUS") or "").strip()
            if ready:
                print(ready.strip())
                print_out(storage.get_text(f"{p}/results/surgery_selftest.txt") or "(no selftest output)", 25)
                return
            if status.startswith(("FAILED", "DONE")):
                print(f"box ended: {status}")
                print_out(storage.get_text(f"{p}/run.log") or "", 40)
                sys.exit(2)
            time.sleep(10)
        sys.exit(f"not ready after {a.wait} min (STATUS {status or 'none'})")

    if a.cmd in ("run", "stop"):
        i = f"{time.time_ns() // 1_000_000:015d}"
        if a.cmd == "stop":
            storage.put_bytes(b"STOP", f"{p}/queue/{i}/cmd")
            print(f"STOP queued ({i}); the box syncs results/ and destroys itself")
            return
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not command:
            sys.exit("no command")
        for spec in a.put:
            local, _, remote = spec.partition("=")
            lp = Path(local)
            lp = lp if lp.is_absolute() else ROOT / lp
            if not lp.is_file():
                sys.exit(f"--put: no file {lp}")
            remote = remote or lp.resolve().relative_to(ROOT).as_posix()
            storage.put(lp, f"{p}/queue/{i}/files/{remote}")
        line = command[0] if len(command) == 1 else " ".join(command)
        storage.put_bytes(line.encode("utf-8"), f"{p}/queue/{i}/cmd")
        print(f"[client] job {i}: {line[:200]}", flush=True)
        sys.exit(wait_result(p, i, a.timeout, a.tail))

    if a.cmd == "result":
        sys.exit(wait_result(p, a.id, a.wait, a.tail))
    if a.cmd == "pull":
        dest = Path(a.local)
        n = storage.download_prefix(f"{p}/results/{a.subdir.strip('/')}", dest)
        print(f"downloaded {n} files to {dest}")
        return
    if a.cmd == "log":
        print_out(storage.get_text(f"{p}/run.log") or "(no log yet)", a.tail)


if __name__ == "__main__":
    main()
