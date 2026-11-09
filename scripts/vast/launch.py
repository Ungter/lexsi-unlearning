"""Run the Phase-1 baseline eval on a Vast.ai GPU; artifacts go to Tigris (bucket lexsi-hackathon).

  python scripts/vast/launch.py up [--offer ID] [--max-dph 2.0] [--min-down 2000] [--eval-args "--tasks calc_direct"] [--dry-run]
  python scripts/vast/launch.py status RUN_ID      instance state + remote STATUS + tail of run.log
  python scripts/vast/launch.py logs RUN_ID        vast container logs (early failures)
  python scripts/vast/launch.py fetch RUN_ID       download to results/runs/RUN_ID/ and merge into results/base/
  python scripts/vast/launch.py down RUN_ID        destroy the instance

Offer choice: single GPU, >= min-down Mbps download, CUDA >= 13.0 (vLLM nightly cu130), verified, reliability > 0.97,
<= max-dph $/h; take the fastest (dlperf) tier and the cheapest offer within 3% of it.
Secrets are read from .env (HF_API, TIGRIS_ACCESS_ID, TIGRIS_SECRET, ENDPOINT) and never printed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import storage  # noqa: E402

BUCKET = "lexsi-hackathon"
# vLLM main gained native LFM3 (Lfm3ForCausalLM) after v0.33.0; pin the nightly that includes it.
IMAGE = "vllm/vllm-openai:nightly-cf5cea2da37d3dfed27d2e0aa3041c3e553c2d57"
DISK_GB = 80
STATE = ROOT / "results" / "vast"


def load_secrets() -> dict:
    from dotenv import dotenv_values

    env = dotenv_values(ROOT / ".env")
    missing = [k for k in ("HF_API", "TIGRIS_ACCESS_ID", "TIGRIS_SECRET", "ENDPOINT") if not env.get(k)]
    if missing:
        sys.exit(f"missing in .env: {missing}")
    ep = env["ENDPOINT"].strip()
    env["ENDPOINT"] = ep if ep.startswith("http") else f"https://{ep}"
    os.environ["AWS_ACCESS_KEY_ID"] = env["TIGRIS_ACCESS_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = env["TIGRIS_SECRET"]
    os.environ["AWS_ENDPOINT_URL_S3"] = env["ENDPOINT"]
    return env


def vast(*args: str, raw: bool = True, confirm: bool = False):
    p = subprocess.run(["vastai", *args, *(["--raw"] if raw else [])], capture_output=True, text=True,
                       input="y\n" if confirm else None)
    if p.returncode:
        sys.exit(f"vastai {' '.join(args[:2])} failed (rc={p.returncode}): {p.stderr.strip()[-400:]}")
    return json.loads(p.stdout) if raw else p.stdout


def pick_offer(max_dph: float, min_down: int) -> dict:
    q = (f"inet_down>={min_down} cuda_max_good>=13.0 num_gpus=1 reliability>0.97 disk_space>={DISK_GB} "
         f"rentable=true verified=true dph_total<={max_dph} gpu_ram>=24")
    offers = vast("search", "offers", q, "-o", "dlperf-")  # results are capped, so sort by speed server-side
    if not offers:
        sys.exit("no offers match")
    best = max(o.get("dlperf") or 0 for o in offers)
    tier = sorted((o for o in offers if (o.get("dlperf") or 0) >= 0.97 * best), key=lambda o: (o["dph_total"], -o["inet_down"]))
    return tier[0]


def offer_brief(o: dict) -> dict:
    return {"offer_id": o["id"], "gpu": o["gpu_name"], "gpu_ram_gb": round(o["gpu_ram"] / 1024), "dph": round(o["dph_total"], 3),
            "inet_down_mbps": int(o["inet_down"]), "dlperf": round(o.get("dlperf") or 0, 1), "cuda": o.get("cuda_max_good"),
            "reliability": round(o.get("reliability2") or 0, 3), "geo": o.get("geolocation")}


def code_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for base in ("src", "scripts"):
            for f in sorted((ROOT / base).rglob("*")):
                if not f.is_file() or "__pycache__" in f.parts or f.suffix not in (".py", ".sh", ".json", ".yaml", ".txt"):
                    continue
                data = f.read_bytes().replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(f.relative_to(ROOT).as_posix())
                info.size, info.mode, info.mtime = len(data), 0o755 if f.suffix == ".sh" else 0o644, int(time.time())
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def state_path(run_id: str) -> Path:
    return STATE / f"{run_id}.json"


def load_state(run_id: str) -> dict:
    p = state_path(run_id)
    if not p.exists():
        sys.exit(f"no state for {run_id} at {p}")
    return json.loads(p.read_text())


def cmd_up(a) -> None:
    sec = load_secrets()
    run_id = time.strftime("base-%Y%m%d-%H%M%S")
    prefix = f"s3://{BUCKET}/runs/{run_id}"
    offer = next((o for o in vast("search", "offers", f"id={a.offer}")), None) if a.offer else pick_offer(a.max_dph, a.min_down)
    if offer is None:
        sys.exit(f"offer {a.offer} not available")
    brief = offer_brief(offer)
    print(json.dumps({"run_id": run_id, "image": IMAGE, **brief}, indent=2))
    if a.dry_run:
        return
    storage.put_bytes(code_tarball(), f"{prefix}/code.tgz")
    url = storage.presign(f"{prefix}/code.tgz", expires=12 * 3600)
    env = {"HF_TOKEN": sec["HF_API"], "AWS_ACCESS_KEY_ID": sec["TIGRIS_ACCESS_ID"], "AWS_SECRET_ACCESS_KEY": sec["TIGRIS_SECRET"],
           "AWS_ENDPOINT_URL_S3": sec["ENDPOINT"], "RUN_ID": run_id, "S3_PREFIX": prefix}
    if "'" in a.eval_args:
        sys.exit("--eval-args must not contain single quotes")
    onstart = (f"mkdir -p /root/lexsi && cd /root/lexsi && curl -fsSL '{url}' -o code.tgz && tar xzf code.tgz && "
               f"(EVAL_ARGS='{a.eval_args}' nohup bash scripts/vast/remote_run.sh > /root/lexsi/run.log 2>&1 &)")
    res = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(DISK_GB),
               "--env", " ".join(f"-e {k}={v}" for k, v in env.items()), "--onstart-cmd", onstart,
               "--ssh", "--direct", "--label", f"lexsi-{run_id}", "--cancel-unavail")
    if not res.get("success"):
        sys.exit("create failed")
    STATE.mkdir(parents=True, exist_ok=True)
    st = {"run_id": run_id, "instance_id": res["new_contract"], "image": IMAGE, "s3_prefix": prefix, "eval_args": a.eval_args,
          "created_unix": int(time.time()), **brief}
    state_path(run_id).write_text(json.dumps(st, indent=2))
    print(f"instance {res['new_contract']} created; state in {state_path(run_id).relative_to(ROOT)}")


def cmd_status(a) -> None:
    load_secrets()
    st = load_state(a.run_id)
    try:
        inst = vast("show", "instance", str(st["instance_id"]))
        keep = ("actual_status", "intended_status", "cur_state", "status_msg", "gpu_name", "dph_total", "start_date", "duration")
        print(json.dumps({k: inst.get(k) for k in keep}, indent=2, default=str))
    except SystemExit as e:
        print(f"instance: {e}")
    print("STATUS:", (storage.get_text(f"{st['s3_prefix']}/STATUS") or "(none yet)").strip())
    log = storage.get_text(f"{st['s3_prefix']}/run.log")
    if log:
        print("\n".join(log.splitlines()[-a.lines:]))


def cmd_logs(a) -> None:
    st = load_state(a.run_id)
    out = vast("logs", str(st["instance_id"]), "--tail", str(a.lines), raw=False)
    print(out[-8000:])


def merge_into_base(run_dir: Path) -> Path:
    """Merge one run's baseline.json/generations into results/base (later runs override per task)."""
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location("baseline_eval", ROOT / "scripts" / "01_baseline_eval.py")
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    base = ROOT / "results" / "base"
    new = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    bpath = base / "baseline.json"
    cur = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else {"model": new["model"], "results": {}, "runs": {}}
    for name, summ in new["results"].items():
        cur["results"][name] = summ | {"run_id": new["run_id"]}
    cur.setdefault("runs", {})[new["run_id"]] = {k: new.get(k) for k in ("env", "args", "load_seconds", "elapsed_seconds")}
    cur["go_no_go"] = be.go_no_go(cur["results"])
    for sub in ("generations", "data"):
        for f in (run_dir / sub).glob("*"):
            (base / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, base / sub / f.name)
    bpath.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return bpath


def cmd_fetch(a) -> None:
    load_secrets()
    st = load_state(a.run_id)
    dest = ROOT / "results" / "runs" / a.run_id
    n = storage.download_prefix(f"{st['s3_prefix']}/results", dest)
    log = storage.get_text(f"{st['s3_prefix']}/run.log")
    if log:
        (dest / "run.log").write_text(log, encoding="utf-8")
    print(f"downloaded {n} files to {dest.relative_to(ROOT)}")
    if (dest / "baseline.json").exists():
        print(f"merged into {merge_into_base(dest).relative_to(ROOT)}")


def cmd_down(a) -> None:
    st = load_state(a.run_id)
    print(vast("destroy", "instance", str(st["instance_id"]), raw=False, confirm=True).strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up")
    up.add_argument("--offer", type=int)
    up.add_argument("--max-dph", type=float, default=2.0)
    up.add_argument("--min-down", type=int, default=2000)
    up.add_argument("--eval-args", default="", help="extra args for scripts/01_baseline_eval.py")
    up.add_argument("--dry-run", action="store_true")
    for name in ("status", "logs", "fetch", "down"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        p.add_argument("--lines", type=int, default=40)
    a = ap.parse_args()
    {"up": cmd_up, "status": cmd_status, "logs": cmd_logs, "fetch": cmd_fetch, "down": cmd_down}[a.cmd](a)


if __name__ == "__main__":
    main()
