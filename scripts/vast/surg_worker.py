"""Phase 7 command worker on a Vast box (started by scripts/vast/jobs/surgery_worker.sh). Lets a local agent drive
surgery sessions on the box without SSH: commands are queued in S3 by scripts/vast/surg_remote.py and run here one at
a time, in /root/lexsi, with bash.

  $S3_PREFIX/queue/<id>/files/<relpath>  files placed under /root/lexsi before the command runs
  $S3_PREFIX/queue/<id>/cmd              the bash command (written last, so a visible cmd means a complete job)
  $S3_PREFIX/done/<id>/out               combined stdout + stderr (also uploaded every 30 s while it runs)
  $S3_PREFIX/done/<id>/rc                exit code (written last)

After each command, new or changed files under results/ are uploaded to $S3_PREFIX/results/. The worker exits on the
command STOP, after IDLE_HOURS without a command or after MAX_HOURS; the job script then syncs, marks DONE and the
instance destroys itself.

  python3 scripts/vast/surg_worker.py MAX_HOURS IDLE_HOURS
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, "src")
from lfm3u import storage  # noqa: E402

PREFIX = os.environ["S3_PREFIX"].rstrip("/")
BUCKET, KEY = storage.split_uri(PREFIX)
CMD_TIMEOUT = 3 * 3600
c = storage.client()
uploaded: dict[str, float] = {}


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def keys(prefix: str) -> list[str]:
    out, tok = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if tok:
            kw["ContinuationToken"] = tok
        r = c.list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def pending() -> list[str]:
    queued = {k.split("/")[-2] for k in keys(f"{KEY}/queue/") if k.endswith("/cmd")}
    done = {k.split("/")[-2] for k in keys(f"{KEY}/done/") if k.endswith("/rc")}
    return sorted(queued - done)


def sync_results() -> int:
    n = 0
    for f in Path("results").rglob("*"):
        if not f.is_file():
            continue
        m = f.stat().st_mtime
        if uploaded.get(f.as_posix()) == m:
            continue
        c.upload_file(str(f), BUCKET, f"{KEY}/{f.as_posix()}")
        uploaded[f.as_posix()] = m
        n += 1
    return n


def run(i: str) -> bool:
    """Run queued job i; returns False when it was STOP."""
    for k in keys(f"{KEY}/queue/{i}/files/"):
        rel = PurePosixPath(k.split(f"/queue/{i}/files/", 1)[1])
        if rel.is_absolute() or ".." in rel.parts:
            print(f"   refusing file path {rel}")
            continue
        Path(rel).parent.mkdir(parents=True, exist_ok=True)
        c.download_file(BUCKET, k, str(rel))
    cmd = storage.get_text(f"{PREFIX}/queue/{i}/cmd") or ""
    if cmd.strip() == "STOP":
        storage.put_bytes(b"STOP received\n", f"{PREFIX}/done/{i}/out")
        storage.put_bytes(b"0\n", f"{PREFIX}/done/{i}/rc")
        return False
    print(f"== {stamp()} cmd {i}: {cmd[:300]}", flush=True)
    out = Path(f"/tmp/out_{i}.txt")
    stop = threading.Event()

    def stream() -> None:  # partial output for long commands
        while not stop.wait(30):
            if out.exists():
                c.upload_file(str(out), BUCKET, f"{KEY}/done/{i}/out")

    th = threading.Thread(target=stream, daemon=True)
    th.start()
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        try:
            rc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, cwd="/root/lexsi",
                                timeout=CMD_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[worker] command timed out after {CMD_TIMEOUT} s\n")
            rc = 124
    stop.set()
    th.join()
    try:
        n = sync_results()
    except Exception as ex:  # keep serving even if a sync fails
        n = f"sync failed: {ex}"
    with out.open("a", encoding="utf-8") as f:
        f.write(f"\n[worker] rc {rc}, {time.time() - t0:.0f} s, results files synced: {n}\n")
    c.upload_file(str(out), BUCKET, f"{KEY}/done/{i}/out")
    storage.put_bytes(f"{rc}\n".encode(), f"{PREFIX}/done/{i}/rc")
    print(f"   rc {rc} in {time.time() - t0:.0f} s", flush=True)
    return True


def main() -> None:
    max_h, idle_h = float(sys.argv[1]), float(sys.argv[2])
    sync_results()
    storage.put_bytes(f"READY {stamp()}\n".encode(), f"{PREFIX}/READY")
    print(f"== {stamp()} worker ready (max {max_h} h, idle {idle_h} h)", flush=True)
    t0 = last = time.time()
    while True:
        if time.time() - t0 > max_h * 3600:
            print(f"== {stamp()} max lifetime reached", flush=True)
            break
        if time.time() - last > idle_h * 3600:
            print(f"== {stamp()} idle timeout", flush=True)
            break
        try:
            ids = pending()
        except Exception as ex:
            print(f"   list failed: {ex}", flush=True)
            time.sleep(10)
            continue
        if not ids:
            time.sleep(4)
            continue
        if not run(ids[0]):
            print(f"== {stamp()} STOP", flush=True)
            break
        last = time.time()
    sync_results()


if __name__ == "__main__":
    main()
