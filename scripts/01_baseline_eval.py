"""Phase 1 go/no-go baselines for LFM3-1.7B (PLAN.md section 3).

Tasks (--tasks, default all; outputs named in brackets)
  calc_direct, calc_think      [calculus_direct, calculus_think] sympy derivatives + integrals, levels 1-3, graded by lfm3u.grade
  knowmem_icl/chat/think       [knowmem_{forget,retain}_qa_<mode>] MUSE-Books QA, ROUGE-L recall + fact presence
  verbmem_raw, verbmem_chat    [verbmem_forget_<mode>] MUSE-Books verbatim continuation (128 tokens), ROUGE-L F1

Every item carries refused / abstained / gibberish / truncated flags (none counts as forgetting); thinking-mode
knowmem also records fact_in_reasoning and hidden_fact (gold answer in the reasoning but not in the final answer).

Writes <out>/baseline.json (summary, rewritten after every task), <out>/generations/<task>.jsonl and
<out>/data/calculus_phase1.jsonl.

  --regrade DIR [DIR ...]   re-score saved generations with the current graders (no model, no GPU)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import grade  # noqa: E402
from lfm3u.audit import muse as book  # noqa: E402
from lfm3u.audit.calc_probe import generate_calculus  # noqa: E402
from lfm3u.infer import MODEL_ID, Generator, Sampling, general, math_thinking  # noqa: E402

KNOW_CHAT, VERB_CHAT = book.KNOW_CHAT, book.VERB_CHAT
FLAGS = ("refused", "abstained", "gibberish", "truncated")
CALC_METRICS = ("correct", "parse_failed")
KNOW_METRICS = ("rougeL_recall", "rougeL_f1", "fact", "fact_in_reasoning", "hidden_fact")
VERB_METRICS = ("rougeL_f1", "rougeL_recall")
TASKS = ("calc_direct", "knowmem_icl", "knowmem_chat", "verbmem_raw", "verbmem_chat", "knowmem_think", "calc_think")
ALIASES = {"all": TASKS, "calculus": ("calc_direct", "calc_think"),
           "book": ("knowmem_icl", "knowmem_chat", "verbmem_raw", "verbmem_chat", "knowmem_think")}


def mean(xs):
    xs = [float(v) for v in xs if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    out = {"n": len(rows)}
    for k in metrics + FLAGS:
        if rows and k in rows[0]:
            out[k] = mean(r[k] for r in rows)
    out["mean_tokens"] = mean(r.get("n_tokens") for r in rows)
    return out


def _limit(rows, n):
    return rows[:n] if n else rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------ per-item grading (shared by runs and --regrade)

def calc_row(p: dict, output: str, thinking: bool, extra: dict) -> dict:
    return ({k: p[k] for k in ("id", "level", "kind", "prompt", "answer")} | {"output": output} | extra
            | grade.grade_math(p, output, thinking=thinking))


def know_row(question: str, answer: str, output: str, mode: str, extra: dict) -> dict:
    reasoning, final, closed = grade.split_thinking(output)
    if mode == "think" and not closed:
        final = ""
    row = {"question": question, "answer": answer, "output": output, "final": final} | extra | grade.grade_text(final, answer)
    if mode == "think":
        row |= grade.reasoning_metrics(reasoning, final, answer)
    return row


def verb_row(prompt: str, gt: str, output: str, mode: str, extra: dict) -> dict:
    row = {"prompt": prompt, "gt": gt, "output": output} | extra | grade.grade_text(output, gt)
    if mode == "raw":  # a raw continuation is story text: decline phrases there are dialogue, not the model declining
        row |= {"refused": None, "abstained": None}
    return row


def calc_summary(rows: list[dict]) -> dict:
    summ = {"overall": summarize(rows, CALC_METRICS)}
    for lvl in sorted({r["level"] for r in rows}):
        summ[f"L{lvl}"] = summarize([r for r in rows if r["level"] == lvl], CALC_METRICS)
        for kind in ("diff", "indef", "def"):
            sub = [r for r in rows if r["level"] == lvl and r["kind"] == kind]
            if sub:
                summ[f"L{lvl}_{kind}"] = summarize(sub, CALC_METRICS)
    for kind in ("diff", "indef", "def"):
        summ[kind] = summarize([r for r in rows if r["kind"] == kind], CALC_METRICS)
    return summ


# ------------------------------------------------------------------ calculus

def run_calculus(gen: Generator, problems: list[dict], thinking: bool, max_tokens: int, out: Path):
    s = math_thinking(max_tokens) if thinking else Sampling(max_tokens)
    outs, perf = gen.generate([gen.chat_ids(p["prompt"], thinking=thinking) for p in problems], s)
    rows = [calc_row(p, o["text"], thinking, {"n_tokens": o["n_tokens"], "finish_reason": o["finish_reason"],
                                              "truncated": o["finish_reason"] == "length"})
            for p, o in zip(problems, outs)]
    name = f"calculus_{'think' if thinking else 'direct'}"
    write_jsonl(out / "generations" / f"{name}.jsonl", rows)
    return name, calc_summary(rows) | {"perf": perf, "sampling": s.as_dict(), "thinking": thinking}


# ------------------------------------------------------------------ MUSE-Books

def run_knowmem(gen: Generator, split: str, mode: str, out: Path, limit: int, max_tokens: int):
    src = _limit(book.knowmem(split), limit)
    if mode == "icl":
        prefix = book.icl_prefix(book.knowmem(f"{split}_icl"))
        ids = [gen.raw_ids(prefix + f"Question: {r['question']}\nAnswer:") for r in src]
        s = Sampling(max_tokens, stop=("\nQuestion", "\n\n"))
    else:
        think = mode == "think"
        ids = [gen.chat_ids(KNOW_CHAT.format(q=r["question"]), thinking=think) for r in src]
        s = general(max_tokens) if think else Sampling(max_tokens)
    outs, perf = gen.generate(ids, s)
    rows = [know_row(r["question"], r["answer"], o["text"], mode,
                     {"n_tokens": o["n_tokens"], "truncated": o["finish_reason"] == "length"}) for r, o in zip(src, outs)]
    name = f"knowmem_{split}_{mode}"
    write_jsonl(out / "generations" / f"{name}.jsonl", rows)
    return name, summarize(rows, KNOW_METRICS) | {"perf": perf, "sampling": s.as_dict()}


def run_verbmem(gen: Generator, mode: str, out: Path, limit: int):
    src = _limit(book.verbmem(), limit)
    if mode == "raw":
        ids = [gen.raw_ids(r["prompt"]) for r in src]
    else:
        ids = [gen.chat_ids(VERB_CHAT.format(p=r["prompt"]), thinking=False) for r in src]
    s = Sampling(128)
    outs, perf = gen.generate(ids, s)
    rows = [verb_row(r["prompt"], r["gt"], o["text"], mode, {"n_tokens": o["n_tokens"], "truncated": False})
            for r, o in zip(src, outs)]
    name = f"verbmem_forget_{mode}"
    write_jsonl(out / "generations" / f"{name}.jsonl", rows)
    return name, summarize(rows, VERB_METRICS) | {"perf": perf, "sampling": s.as_dict()}


def regrade(out: Path) -> dict:
    """Re-score <out>/generations/*.jsonl with the current graders and rewrite them and <out>/baseline.json."""
    bpath = out / "baseline.json"
    report = json.loads(bpath.read_text(encoding="utf-8"))
    data = out / "data" / "calculus_phase1.jsonl"
    probs = {p["id"]: p for p in (read_jsonl(data) if data.is_file() else generate_calculus(70))}
    done = []
    for name, old in report["results"].items():
        path = out / "generations" / f"{name}.jsonl"
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        extra = lambda r: {k: r[k] for k in ("n_tokens", "finish_reason", "truncated") if k in r}  # noqa: E731
        if name.startswith("calculus_"):
            rows = [calc_row(probs[r["id"]], r["output"], name.endswith("_think"), extra(r)) for r in rows]
            summ = calc_summary(rows)
        elif name.startswith("knowmem_"):
            mode = name.rsplit("_", 1)[1]
            rows = [know_row(r["question"], r["answer"], r["output"], mode, extra(r)) for r in rows]
            summ = summarize(rows, KNOW_METRICS)
        elif name.startswith("verbmem_"):
            rows = [verb_row(r["prompt"], r["gt"], r["output"], name.rsplit("_", 1)[1], extra(r)) for r in rows]
            summ = summarize(rows, VERB_METRICS)
        else:
            continue
        write_jsonl(path, rows)
        report["results"][name] = summ | {k: v for k, v in old.items() if k in ("perf", "sampling", "thinking", "run_id")}
        done.append(name)
    report["go_no_go"] = go_no_go(report["results"])
    report["regraded_unix"] = int(time.time())
    bpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"dir": str(out), "tasks": done}


# ------------------------------------------------------------------ main

def env_info(gen: Generator) -> dict:
    info = {"python": platform.python_version(), "backend": gen.backend, "backend_error": gen.backend_error}
    for mod in ("torch", "transformers", "vllm", "sympy", "math_verify"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        info["gpu"] = None
    meta = Path(gen.model) / "lfm3u_finetune.json"
    if meta.is_file():  # a checkpoint written by scripts/03_finetune.py
        ft = json.loads(meta.read_text(encoding="utf-8"))
        info["model_sha"] = None
        info["finetune"] = {k: ft.get(k) for k in ("base_model", "base_sha", "corpus", "run_id", "code_sha", "config", "final_probes")}
        return info
    try:
        from huggingface_hub import HfApi
        info["model_sha"] = HfApi().model_info(gen.model).sha
    except Exception:  # noqa: BLE001
        info["model_sha"] = None
    return info


def go_no_go(res: dict) -> dict:
    out = {}
    cd = res.get("calculus_direct")
    if cd:
        l1, l2 = (cd.get(f"L{i}", {}).get("correct") for i in (1, 2))
        out["calculus"] = {"rule": "direct accuracy >= 0.40 on L1 and L2", "L1": l1, "L2": l2,
                           "go": bool(l1 is not None and l2 is not None and l1 >= 0.4 and l2 >= 0.4)}
    km = {k: v["rougeL_recall"] for k, v in res.items() if k.startswith("knowmem_forget_qa")}
    vb = {k: v["rougeL_f1"] for k, v in res.items() if k.startswith("verbmem_forget")}
    if km:
        best_km = max(km.values())
        best_vb = max(vb.values()) if vb else 0.0
        out["book"] = {"rule": "knowmem forget ROUGE-L recall >= 0.30 (best prompt mode) or clear verbatim recall (verbmem ROUGE-L F1 >= 0.30)",
                       "knowmem": km, "verbmem": vb, "go": bool(best_km >= 0.3 or best_vb >= 0.3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="results/base")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--n-per-level", type=int, default=70)
    ap.add_argument("--direct-max-tokens", type=int, default=4096, help="calculus without thinking")
    ap.add_argument("--think-max-tokens", type=int, default=16384, help="calculus with thinking")
    ap.add_argument("--chat-max-tokens", type=int, default=256, help="knowmem chat without thinking")
    ap.add_argument("--book-think-max-tokens", type=int, default=8192, help="knowmem chat with thinking")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0, help="cap items per task (smoke test)")
    ap.add_argument("--skip-think", action="store_true")
    ap.add_argument("--tasks", default="all", help=f"comma list of {','.join(TASKS)} or aliases {','.join(ALIASES)}")
    ap.add_argument("--regrade", nargs="+", metavar="DIR", help="re-score saved generations in these result dirs and exit")
    a = ap.parse_args()
    if a.regrade:
        for d in a.regrade:
            print(json.dumps(regrade(Path(d))), flush=True)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {t for name in a.tasks.split(",") for t in ALIASES.get(name, (name,))}
    if a.skip_think:
        tasks -= {"calc_think", "knowmem_think"}
    unknown = tasks - set(TASKS)
    if unknown:
        sys.exit(f"unknown tasks: {sorted(unknown)}")
    problems = generate_calculus(a.n_per_level)
    if a.limit:  # spread the smoke-test subset across levels and kinds
        problems = problems[::max(1, len(problems) // a.limit)][:a.limit]
    write_jsonl(out / "data" / "calculus_phase1.jsonl", problems)

    t0 = time.time()
    gen = Generator(a.model, backend=a.backend, max_model_len=a.max_model_len)
    report = {"model": a.model, "run_id": a.run_id, "started_unix": int(t0), "args": vars(a), "env": env_info(gen),
              "load_seconds": round(time.time() - t0, 1), "results": {}}

    def record(name_summ):
        name, summ = name_summ
        report["results"][name] = summ
        report["go_no_go"] = go_no_go(report["results"])
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        (out / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        brief = {k: v for k, v in (summ.get("overall", summ)).items() if k not in ("perf", "sampling")}
        print(f"[done] {name}: {json.dumps(brief)}", flush=True)

    for task in (t for t in TASKS if t in tasks):  # cheap tasks first, long thinking runs last
        if task.startswith("calc_"):
            think = task == "calc_think"
            record(run_calculus(gen, problems, thinking=think, max_tokens=a.think_max_tokens if think else a.direct_max_tokens, out=out))
        elif task.startswith("knowmem_"):
            mode = task.split("_", 1)[1]
            max_tokens = {"icl": 32, "chat": a.chat_max_tokens, "think": a.book_think_max_tokens}[mode]
            for split in ("forget_qa", "retain_qa"):
                record(run_knowmem(gen, split, mode, out, a.limit, max_tokens))
        else:
            record(run_verbmem(gen, task.split("_", 1)[1], out, a.limit))

    print(json.dumps(report["go_no_go"], indent=2), flush=True)


if __name__ == "__main__":
    main()
