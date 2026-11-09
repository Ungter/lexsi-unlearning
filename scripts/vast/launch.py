"""Run jobs on a Vast.ai GPU; artifacts go to Tigris (bucket lexsi-hackathon).

  python scripts/vast/launch.py up [--job JOB[+JOB...]] [--offer ID] [--max-dph 2.0] [--min-down 2000]
                                   [--eval-args "--tasks calc_direct"] [--ft-args "--epochs 8"]
                                   [--model-uri s3://lexsi-hackathon/models/... --eval-name NAME]
                                   [--models name=REF,name=REF] [--job-script PATH[,PATH]] [--script-args "..."]
                                   [--replay-uri s3://lexsi-hackathon/runs/RUN_ID/work/replay.jsonl] [--dry-run]
      baseline: scripts/01_baseline_eval.py on LFM3-1.7B (or on --model-uri) -> results/<name>/
      eval:     scripts/01_baseline_eval.py on every --models entry (REF = s3:// checkpoint prefix or HF id)
      finetune: self-replay, fine-tune the MUSE target and retain-only models (scripts/02_replay.py, 03_finetune.py),
                upload them to s3://lexsi-hackathon/models/RUN_ID/, then evaluate both -> results/{target,retain}/
      script:   run --job-script (bash, from the repo; may source scripts/vast/common.sh) with --script-args
    Jobs chain with "+", e.g. --job eval+script, and share one instance.
  python scripts/vast/launch.py status RUN_ID      instance state + remote STATUS + tail of run.log
  python scripts/vast/launch.py logs RUN_ID        vast container logs (early failures)
  python scripts/vast/launch.py fetch RUN_ID       download to results/runs/RUN_ID/ and merge into results/<name>/
  python scripts/vast/launch.py down RUN_ID        destroy the instance

Offer choice: single GPU, >= min-down Mbps download, CUDA >= 13.0 (vLLM nightly cu130), verified, reliability > 0.97,
<= max-dph $/h; take the fastest (dlperf) tier and the cheapest offer within 3% of it.
up waits until the container is running; a host still 'loading' after --load-timeout min (stalled image pull) is
destroyed and another machine is picked (--retries). Instances destroy themselves when the job ends (DONE or FAILED,
after results and run.log are synced) unless --keep; down then just reports the instance is gone.
Secrets are read from .env (HF_API, TIGRIS_ACCESS_ID, TIGRIS_SECRET, ENDPOINT) and never printed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
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
JOBS = ("baseline", "eval", "finetune", "script")


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
    if not raw:
        return p.stdout
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        sys.exit(f"vastai {' '.join(args[:2])}: unexpected output: {(p.stdout or p.stderr).strip()[-400:]}")
    if isinstance(out, dict) and out.get("error"):
        sys.exit(f"vastai {' '.join(args[:2])}: {out.get('msg') or out}")
    return out


def pick_offer(max_dph: float, min_down: int, exclude_machines=()) -> dict:
    q = (f"inet_down>={min_down} cuda_max_good>=13.0 num_gpus=1 reliability>0.97 disk_space>={DISK_GB} "
         f"rentable=true verified=true dph_total<={max_dph} gpu_ram>=24")
    offers = vast("search", "offers", q, "-o", "dlperf-")  # results are capped, so sort by speed server-side
    offers = [o for o in offers if o.get("machine_id") not in set(exclude_machines)]
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


def code_sha() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts") else "")


def cmd_up(a) -> None:
    sec = load_secrets()
    jobs = a.job.split("+")
    if not all(j in JOBS for j in jobs):
        sys.exit(f"--job must be {'|'.join(JOBS)} or several joined with +")
    if "eval" in jobs and not a.models:
        sys.exit("--job eval needs --models")
    if "script" in jobs and not a.job_script:
        sys.exit("--job script needs --job-script")
    for s in (a.job_script or "").split(","):
        if s and not (ROOT / s).is_file():
            sys.exit(f"job script not found: {s}")
    prefix_name = {"baseline": "base", "finetune": "ft"}.get(jobs[0], jobs[0])
    run_id = time.strftime(f"{prefix_name}-%Y%m%d-%H%M%S")
    prefix = f"s3://{BUCKET}/runs/{run_id}"
    if a.offer:  # the CLI warns about ask_contract_id but the API filters on it ("id=" matches nothing)
        offer = next((o for o in vast("search", "offers", f"ask_contract_id={a.offer}") if o["id"] == a.offer), None)
    else:
        offer = pick_offer(a.max_dph, a.min_down)
    if offer is None:
        sys.exit(f"offer {a.offer} not available")
    print(json.dumps({"run_id": run_id, "image": IMAGE, **offer_brief(offer)}, indent=2))
    if a.dry_run:
        return
    storage.put_bytes(code_tarball(), f"{prefix}/code.tgz")
    url = storage.presign(f"{prefix}/code.tgz", expires=12 * 3600)
    env = {"HF_TOKEN": sec["HF_API"], "AWS_ACCESS_KEY_ID": sec["TIGRIS_ACCESS_ID"], "AWS_SECRET_ACCESS_KEY": sec["TIGRIS_SECRET"],
           "AWS_ENDPOINT_URL_S3": sec["ENDPOINT"], "RUN_ID": run_id, "S3_PREFIX": prefix, "CODE_SHA": code_sha(), "JOB": a.job,
           "AUTO_DESTROY": "0" if a.keep else "1"}
    if a.model_uri:
        env |= {"MODEL_URI": a.model_uri, "EVAL_NAME": a.eval_name}
    if a.replay_uri:
        env["REPLAY_URI"] = a.replay_uri
    if a.models:
        env["EVAL_MODELS"] = a.models
    if a.job_script:
        env["JOB_SCRIPT"] = a.job_script
    if any(c in s for s in (a.eval_args, a.ft_args, a.script_args) for c in "'\"\\$`") or not all(
            re.fullmatch(r"[\w./:=,+-]*", s) for s in (a.model_uri or "", a.eval_name, a.replay_uri or "",
                                                       a.models or "", a.job_script or "")):
        sys.exit("--eval-args/--ft-args/--script-args must not contain quotes, backslashes, $ or backticks; "
                 "--model-uri/--eval-name/--models/--job-script must be plain paths")
    onstart = (f"mkdir -p /root/lexsi && cd /root/lexsi && curl -fsSL '{url}' -o code.tgz && tar xzf code.tgz && "
               f"(EVAL_ARGS='{a.eval_args}' FT_ARGS='{a.ft_args}' SCRIPT_ARGS='{a.script_args}' "
               "nohup bash scripts/vast/remote_run.sh > /root/lexsi/run.log 2>&1 &)")
    STATE.mkdir(parents=True, exist_ok=True)
    stalled: list[int] = []
    for attempt in range(a.retries + 1):
        if attempt:  # the previous host stalled: re-pick automatically, skipping the machines that stalled
            offer = pick_offer(a.max_dph, a.min_down, exclude_machines=stalled)
            print(json.dumps({"retry": attempt, **offer_brief(offer)}, indent=2))
        res = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(DISK_GB),
                   "--env", " ".join(f"-e {k}={v}" for k, v in env.items()), "--onstart-cmd", onstart,
                   "--ssh", "--direct", "--label", f"lexsi-{run_id}", "--cancel-unavail")
        if not res.get("success"):
            sys.exit("create failed")
        iid = res["new_contract"]
        st = {"run_id": run_id, "job": a.job, "instance_id": iid, "machine_id": offer.get("machine_id"), "image": IMAGE,
              "s3_prefix": prefix, "code_sha": env["CODE_SHA"], "eval_args": a.eval_args, "ft_args": a.ft_args,
              "model_uri": a.model_uri, "replay_uri": a.replay_uri, "models": a.models, "job_script": a.job_script,
              "script_args": a.script_args, "auto_destroy": not a.keep, "stalled_machines": stalled,
              "created_unix": int(time.time()), **offer_brief(offer)}
        state_path(run_id).write_text(json.dumps(st, indent=2))
        print(f"instance {iid} created; state in {state_path(run_id).relative_to(ROOT)}", flush=True)
        if not a.load_timeout or wait_running(iid, a.load_timeout * 60):
            return
        print(f"instance {iid} still loading after {a.load_timeout} min; destroying it and re-picking", flush=True)
        vast("destroy", "instance", str(iid), raw=False, confirm=True)
        stalled.append(offer.get("machine_id"))
    sys.exit(f"gave up after {a.retries + 1} hosts stalled while loading")


def wait_running(instance_id: int, timeout_s: float) -> bool:
    """Poll until the container is running (True) or timeout_s passes (False). Healthy hosts take 1-6 min;
    stalled image pulls sit in 'loading' indefinitely and are billed."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            inst = vast("show", "instance", str(instance_id))
        except SystemExit:
            inst = {}
        if inst.get("actual_status") == "running":
            print(f"instance {instance_id} running after {(time.time() - t0) / 60:.1f} min", flush=True)
            return True
        if inst.get("actual_status") in ("exited", "offline"):
            return False
        time.sleep(20)
    return False


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


def merge_baseline(src: Path, base: Path) -> Path:
    """Merge one run's baseline.json/generations (in src) into base (later runs override per task)."""
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location("baseline_eval", ROOT / "scripts" / "01_baseline_eval.py")
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    run_dir = src
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
    import shutil

    for f in sorted(dest.rglob("baseline.json")):
        name = "base" if f.parent == dest else f.parent.relative_to(dest).as_posix()  # flat layout = early baseline runs
        print(f"merged into {merge_baseline(f.parent, ROOT / 'results' / name).relative_to(ROOT)}")
    if (dest / "finetune").is_dir():
        shutil.copytree(dest / "finetune", ROOT / "results" / "finetune" / a.run_id, dirs_exist_ok=True)
        print(f"copied fine-tune logs to results/finetune/{a.run_id}")


def cmd_down(a) -> None:
    st = load_state(a.run_id)
    try:
        out = vast("destroy", "instance", str(st["instance_id"]), raw=False, confirm=True).strip()
    except SystemExit as e:
        out = f"instance {st['instance_id']} already gone (auto-destroyed?): {e}"
    print(out or f"instance {st['instance_id']} already gone (auto-destroyed?)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up")
    up.add_argument("--offer", type=int)
    up.add_argument("--max-dph", type=float, default=2.0)
    up.add_argument("--min-down", type=int, default=2000)
    up.add_argument("--job", default="baseline", help=f"{'|'.join(JOBS)}, or several joined with +")
    up.add_argument("--models", default=None, help="eval job: name=REF,name=REF (REF = s3:// prefix or HF id)")
    up.add_argument("--job-script", default=None, help="script job: bash script(s) in the repo, comma-separated")
    up.add_argument("--script-args", default="", help="script job: arguments for the script(s)")
    up.add_argument("--eval-args", default="", help="extra args for scripts/01_baseline_eval.py")
    up.add_argument("--ft-args", default="", help="extra args for scripts/03_finetune.py")
    up.add_argument("--model-uri", default=None, help="baseline job: evaluate this checkpoint prefix instead of LFM3-1.7B")
    up.add_argument("--eval-name", default="eval", help="baseline job with --model-uri: results/<name>/")
    up.add_argument("--replay-uri", default=None, help="finetune job: reuse this self-replay jsonl (s3://...)")
    up.add_argument("--keep", action="store_true", help="do not let the instance destroy itself when the job ends")
    up.add_argument("--load-timeout", type=float, default=10.0,
                    help="minutes to wait for the container to start before destroying the host and re-picking (0 = don't wait)")
    up.add_argument("--retries", type=int, default=2, help="re-picks after stalled hosts")
    up.add_argument("--dry-run", action="store_true")
    for name in ("status", "logs", "fetch", "down"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        p.add_argument("--lines", type=int, default=40)
    a = ap.parse_args()
    {"up": cmd_up, "status": cmd_status, "logs": cmd_logs, "fetch": cmd_fetch, "down": cmd_down}[a.cmd](a)


if __name__ == "__main__":
    main()
