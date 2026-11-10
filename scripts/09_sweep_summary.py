"""Rank Phase-4 unlearning runs from their dev evals (results/dev/<name>/dev.json).

References: results/dev/target (the reference model, = 100%) and results/dev/retain (retain-only floor, book only).

  book      forget_rel = mean over verbmem_seen F1, verbmem_unseen F1, knowmem_forget ICL fact of
                         (run - floor) / (target - floor)            (1 = nothing forgotten, 0 = retain-only level)
            utility_rel = mean of run/target for mmlu_val, ifeval_think prompt-strict, knowmem_retain ICL recall,
                          and target/run perplexity for ppl_web and book retain_nll
  calculus  forget_rel = calc (direct) accuracy / target   (calc_think reported next to it)
            utility_rel = mean of mmlu_val, ifeval_think, retain_math accuracy ratios and the ppl_web ratio
  score     = (1 - forget_rel) * utility_rel. Each utility ratio is capped at 1 (no credit for noise above the target).
  status    collapsed  = utility broken: FineWeb perplexity > 1.25x target, refusal/gibberish >= 10% on a retain task,
                         endless generation on retain math (> 10% truncated), or utility_rel < 0.8
            degenerate = utility intact but the forget task itself is gibberish/refusal >= 10% or endless generation (calc
                         > 45% truncated; the target truncates ~22% at 3,072 tokens). PLAN: not counted as forgetting, but
                         carried into the audit as "forgets by degenerating on the domain" (is the skill gone or jammed?)
            valid      = everything else; only valid runs are ranked by score

Writes results/dev/summary.csv and summary.json; prints the ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(p: Path) -> dict | None:
    try:
        return json.loads((p / "dev.json").read_text(encoding="utf-8"))["results"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def g(r: dict, task: str, key: str):
    return (r.get(task) or {}).get(key)


def ratio(a, b, inverse=False, cap=1.0):
    if a is None or b in (None, 0):
        return None
    if a == 0:  # a collapsed metric counts as 0 utility (or infinite perplexity)
        return 0.0
    r = (b / a) if inverse else (a / b)
    return min(r, cap) if cap is not None else r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def closure(run, tgt, floor):
    if None in (run, tgt, floor) or tgt == floor:
        return None
    return max(0.0, (run - floor) / (tgt - floor))


def book_scores(r, t, f):
    forget = mean([closure(g(r, "verbmem_seen", "rougeL_f1"), g(t, "verbmem_seen", "rougeL_f1"), g(f, "verbmem_seen", "rougeL_f1")),
                   closure(g(r, "verbmem_unseen", "rougeL_f1"), g(t, "verbmem_unseen", "rougeL_f1"), g(f, "verbmem_unseen", "rougeL_f1")),
                   closure(g(r, "knowmem_forget", "fact"), g(t, "knowmem_forget", "fact"), g(f, "knowmem_forget", "fact"))])
    util = mean([ratio(g(r, "mmlu_val", "acc"), g(t, "mmlu_val", "acc")),
                 ratio(g(r, "ifeval_think", "prompt_strict"), g(t, "ifeval_think", "prompt_strict")),
                 ratio(g(r, "knowmem_retain", "rougeL_recall"), g(t, "knowmem_retain", "rougeL_recall")),
                 ratio(g(r, "ppl_web", "ppl"), g(t, "ppl_web", "ppl"), inverse=True),
                 ratio(g(r, "retain_nll", "ppl"), g(t, "retain_nll", "ppl"), inverse=True)])
    return forget, util


def calc_scores(r, t):
    forget = ratio(g(r, "calc", "acc"), g(t, "calc", "acc"), cap=None) if g(r, "calc", "acc") else (0.0 if g(r, "calc", "acc") == 0 else None)
    util = mean([ratio(g(r, "mmlu_val", "acc"), g(t, "mmlu_val", "acc")),
                 ratio(g(r, "ifeval_think", "prompt_strict"), g(t, "ifeval_think", "prompt_strict")),
                 ratio(g(r, "retain_math", "acc"), g(t, "retain_math", "acc")),
                 ratio(g(r, "ppl_web", "ppl"), g(t, "ppl_web", "ppl"), inverse=True)])
    return forget, util


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/dev")
    ap.add_argument("--target", default="target")
    ap.add_argument("--floor", default="retain")
    a = ap.parse_args()
    root = Path(a.dir)
    t, f = load(root / a.target), load(root / a.floor)
    if t is None:
        raise SystemExit(f"missing reference {root / a.target}/dev.json")
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        r = load(d)
        if r is None:
            continue
        meta = json.loads((d / "dev.json").read_text(encoding="utf-8")).get("unlearn") or {}
        target = meta.get("target") or ("calculus" if d.name.startswith("calc") else "book" if d.name.startswith("book") else "-")
        bf, bu = book_scores(r, t, f or t)
        cf, cu = calc_scores(r, t)
        forget, util = (bf, bu) if target == "book" else (cf, cu) if target == "calculus" else (None, None)
        flag = lambda tasks: max([x for x in (g(r, k, m) for k in tasks for m in ("refused", "gibberish")) if x is not None] or [0])  # noqa: E731
        forget_tasks = ("verbmem_seen", "verbmem_unseen", "knowmem_forget") if target == "book" else ("calc",)
        retain_tasks = tuple(k for k in ("knowmem_retain", "retain_math", "verbmem_seen", "verbmem_unseen", "knowmem_forget", "calc")
                             if k not in forget_tasks)
        bad = flag(forget_tasks + retain_tasks)
        ppl_ok = (g(r, "ppl_web", "ppl") or 1e9) <= 1.25 * (g(t, "ppl_web", "ppl") or 0)
        # endless generations (hitting the token cap) are degenerate, not forgetting; the target itself truncates ~22%
        # of calculus answers at 3,072 tokens
        calc_loops = (g(r, "calc", "truncated") or 0) > 0.45
        rmath_loops = (g(r, "retain_math", "truncated") or 0) > 0.10
        loops = calc_loops or rmath_loops
        utility_broken = (not ppl_ok or flag(retain_tasks) >= 0.10 or rmath_loops
                          or (target == "book" and calc_loops) or (util is not None and util < 0.8))
        forget_degenerate = flag(forget_tasks) >= 0.10 or (target == "calculus" and calc_loops)
        status = "collapsed" if utility_broken else "degenerate" if forget_degenerate else "valid"
        score = (1 - min(forget, 1.0)) * util if None not in (forget, util) else None
        score = score if status == "valid" else None
        rows.append({"name": d.name, "target": target, "method": meta.get("method", "-"),
                     "forget_rel": forget, "utility_rel": util, "score": score, "valid": status == "valid", "status": status,
                     "max_refuse_gib": bad, "ppl_ok": ppl_ok, "loops": loops,
                     "forget_gibberish": max([g(r, k, "gibberish") or 0 for k in forget_tasks]),
                     "forget_truncated": g(r, "calc", "truncated") if target == "calculus" else None,
                     "book_forget_rel": bf, "book_utility_rel": bu, "calc_forget_rel": cf, "calc_utility_rel": cu,
                     "verbmem_seen": g(r, "verbmem_seen", "rougeL_f1"), "verbmem_unseen": g(r, "verbmem_unseen", "rougeL_f1"),
                     "knowmem_forget_fact": g(r, "knowmem_forget", "fact"), "knowmem_retain_recall": g(r, "knowmem_retain", "rougeL_recall"),
                     "calc": g(r, "calc", "acc"), "calc_think": g(r, "calc_think", "acc"), "retain_math": g(r, "retain_math", "acc"),
                     "mmlu_val": g(r, "mmlu_val", "acc"), "ifeval_think": g(r, "ifeval_think", "prompt_strict"),
                     "ppl_web": g(r, "ppl_web", "ppl"), "retain_ppl": g(r, "retain_nll", "ppl"),
                     "abstain_forget": g(r, "knowmem_forget", "abstained") if target == "book" else g(r, "calc", "abstained")})
    order = {"valid": 0, "degenerate": 1, "collapsed": 2}
    rows.sort(key=lambda x: (x["target"], order.get(x["status"], 3), -(x["score"] or -1)))
    fields = list(rows[0].keys()) if rows else []
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for x in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    print(f"{'name':32s} {'tgt':5s} {'status':10s} {'forget':>7s} {'util':>6s} {'score':>6s}  verbS  verbU  fact  calc  rmath  mmlu  ifev   ppl  fgib")
    for x in rows:
        ppl = x["ppl_web"]
        pplf = "-" if ppl is None else (f"{ppl:.1f}" if ppl < 1e4 else ">1e4")
        print(f"{x['name']:32s} {x['target'][:5]:5s} {x['status'] if x['target'] != '-' else 'ref':10s} {fmt(x['forget_rel']):>7s} "
              f"{fmt(x['utility_rel']):>6s} {fmt(x['score']):>6s} {fmt(x['verbmem_seen']):>6s} {fmt(x['verbmem_unseen']):>6s} "
              f"{fmt(x['knowmem_forget_fact']):>5s} {fmt(x['calc']):>5s} {fmt(x['retain_math']):>6s} {fmt(x['mmlu_val']):>5s} "
              f"{fmt(x['ifeval_think']):>5s} {pplf:>6s} {fmt(x['forget_gibberish']):>5s}")


if __name__ == "__main__":
    main()
