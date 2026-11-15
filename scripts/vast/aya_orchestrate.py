"""Unattended launcher for the Tiny Aya pipeline after the reference refit (run locally as a background job).

  python scripts/vast/aya_orchestrate.py TARGET_RUN RETAIN_RUN

1. target uploaded (refit run.log reaches "eval aya-muse-target")  -> wave 1 (scripts/vast/waves/w1.txt)
2. retain-only uploaded                                            -> wave 2 (w2.txt: reference audits / attacks)
3. target refit DONE (its baseline eval wrote the knowmem generations) -> J-space questions file, jspace box
   (MODELS=base,target,retain), then the surgery-bundle box (waits for selfgen, localizer and the target lens)
Every launch is logged to results/vast/orchestrate.log and printed.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "vast"), str(ROOT / "src")]

import launch  # noqa: E402
import wave  # noqa: E402
from lfm3u import storage  # noqa: E402

BUCKET = "s3://lexsi-hackathon"
LOG = ROOT / "results/vast/orchestrate.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_log(run_id: str) -> tuple[str, str]:
    st = launch.load_state(run_id)
    return (storage.get_text(f"{st['s3_prefix']}/STATUS") or "").strip(), storage.get_text(f"{st['s3_prefix']}/run.log") or ""


def wait_for(run_id: str, pattern: str | None, done: bool = False, every: int = 60) -> str:
    while True:
        status, text = run_log(run_id)
        if status.startswith("FAILED"):
            log(f"{run_id} FAILED: {status}")
            sys.exit(1)
        if (done and status.startswith("DONE")) or (pattern and re.search(pattern, text)):
            return text
        time.sleep(every)


def launch_wave(path: str) -> list[dict]:
    lines = [ln.strip() for ln in (ROOT / path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    out = []
    for ln in lines:
        r = wave.run(ln)
        out.append(r)
        log(f"{'OK ' if r['ok'] else 'ERR'} {r['run_id']} {r['gpu']} | {ln[:100]}" + ("" if r["ok"] else f" | {r['tail']}"))
        time.sleep(2)
    return out


def main() -> None:
    t_run, r_run = sys.argv[1], sys.argv[2]
    launch.load_secrets()
    T = f"{BUCKET}/models/{t_run}/aya-muse-target-A15"
    R = f"{BUCKET}/models/{r_run}/aya-muse-retain-A15"
    os.environ |= {"T": T, "R": R}
    log(f"orchestrator start: T={T} R={R}")

    wait_for(t_run, r"== \S+ eval aya-muse-target")
    log("target uploaded -> wave 1")
    launch_wave("scripts/vast/waves/w1.txt")

    wait_for(r_run, r"== \S+ eval aya-muse-retain")
    log("retain-only uploaded -> wave 2")
    launch_wave("scripts/vast/waves/w2.txt")

    wait_for(t_run, None, done=True)
    st = launch.load_state(t_run)
    gen = storage.get_text(f"{st['s3_prefix']}/results/target_A15/generations/knowmem_forget_qa_chat.jsonl")
    if not gen:
        log("no knowmem_forget_qa_chat generations for the target; J-space and surgery bundle NOT launched")
        sys.exit(1)
    p = ROOT / "work/jspace_inputs/knowmem_forget_qa_chat.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(gen, encoding="utf-8")
    from lfm3u.jspace.validate import build_questions

    rows = build_questions(p, p)
    q = p.with_name("knowmem_forget_qa_jspace.jsonl")
    q.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    storage.put(q, f"{BUCKET}/aya/jspace/inputs/knowmem_forget_qa_jspace.jsonl")
    log(f"J-space questions: {len(rows)} rows uploaded")
    js = wave.run(f'--job script --job-script scripts/vast/jobs/jspace.sh --script-args "MODELS=target,base,retain SMOKE=0 '
                  f'TARGET_URI={T} RETAIN_URI={R}" --tag jspace --min-gpu-ram 90 --max-dph 2.4')
    log(f"{'OK ' if js['ok'] else 'ERR'} {js['run_id']} {js['gpu']} | jspace | {js['tail'] if not js['ok'] else ''}")
    if not js["ok"]:
        sys.exit(1)
    lens = f"{BUCKET}/aya/jspace/{js['run_id']}/target"
    sb = wave.run(f"--job script --job-script scripts/vast/jobs/surgery_bundle.sh --tag sbundle --min-gpu-ram 40 --max-dph 2.4 "
                  f"--extra-env TARGET_URI={T},LENS_URI={lens}")
    log(f"{'OK ' if sb['ok'] else 'ERR'} {sb['run_id']} {sb['gpu']} | surgery bundle (lens {lens}) | {sb['tail'] if not sb['ok'] else ''}")
    log("orchestrator done")


if __name__ == "__main__":
    main()
