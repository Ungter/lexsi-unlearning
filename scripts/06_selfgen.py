"""Self-generated training targets for Phase-4 unlearning: the TARGET model's own answers (token ids).

  calculus     calculus/level{1,2,3}/train prompts; direct mode (greedy) for all; only correctly graded, untruncated
               answers kept. Tiny Aya has no thinking mode, so --think-frac now defaults to 0: a non-zero value adds
               a sampled (model-card math sampling) second answer for that fraction, same prompt, tagged mode "think"
               only for compatibility
  retain_math  arithmetic/algebra/precalculus level{1,2,3}/train, direct mode; correct answers only. The math forget
               target of this branch (arch.MATH_TARGET = algebra) is sampled with --target-per-level (default 300, so
               ~500 correct algebra rows exist for --n-forget 400), the other topics with --retain-per-level. Calculus
               keeps source 'calculus' (now a RETAIN topic on the Aya branch).
  general      general/retain_train (cleaned Alpaca) instructions, direct mode, model-card general sampling;
               untruncated, non-empty, non-gibberish answers

Forgetting the model's own reasoning traces is the faithful calculus target, and retaining its own style avoids the
style shift that cross-entropy on terse "Answer: <expr>" labels caused in the SafeTune smoke test. Train/dev splits
only (--split dev writes the dev-split version used by the relearning and in-context attacks). Writes <out> (jsonl: id, source, topic, level, mode, prompt_ids, completion_ids) and <out>.summary.json.
Completions are closed with <|END_RESPONSE|><|END_OF_TURN_TOKEN|> (arch.answer_suffix()) whatever stop token the
generator ended on, so unlearning losses cover the turn end like the fine-tune did.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import arch, grade  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402
from lfm3u.infer import Generator, Sampling, general, math_thinking  # noqa: E402

RETAIN_TOPICS = ("arithmetic", "algebra", "precalculus")


def close_turn(comp: list[int], stop: set[int], suffix: list[int]) -> list[int]:
    """Strip trailing stop tokens and append the template's end tokens."""
    while comp and comp[-1] in stop:
        comp = comp[:-1]
    return comp + suffix if comp else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train", choices=("train", "dev"), help="dev = relearning/ICL attack data (PLAN 6.3 #9, #10)")
    ap.add_argument("--retain-topics", default=",".join(RETAIN_TOPICS))
    ap.add_argument("--out", default="work/selfgen.jsonl")
    ap.add_argument("--calc-per-level", type=int, default=300)
    ap.add_argument("--think-frac", type=float, default=0.0,
                    help="extra sampled calculus answers per level (LFM3 thinking mode; Tiny Aya has none)")
    ap.add_argument("--retain-per-level", type=int, default=100, help="per topic and level (topics other than arch.MATH_TARGET)")
    ap.add_argument("--target-per-level", type=int, default=300,
                    help=f"per level for the math forget target arch.MATH_TARGET={arch.MATH_TARGET} (when it is a retain_math topic)")
    ap.add_argument("--general", type=int, default=800)
    ap.add_argument("--max-len", type=int, default=4096, help="max prompt + completion tokens kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap every set (smoke test)")
    a = ap.parse_args()
    a.max_len = min(a.max_len, arch.MAX_CONTEXT)

    t0 = time.time()
    data = Path(a.data)
    rng = random.Random(a.seed)
    cap = lambda n: min(n, a.limit) if a.limit else n  # noqa: E731

    def sample(path: Path, n: int) -> list[dict]:
        rows = read_jsonl(path)
        rng.shuffle(rows)
        return rows[:cap(n)]

    calc = {lvl: sample(data / f"calculus/level{lvl}/{a.split}.jsonl", a.calc_per_level) for lvl in (1, 2, 3)}
    per_topic = lambda t: a.target_per_level if t == arch.MATH_TARGET else a.retain_per_level  # noqa: E731
    retain = [r for t in a.retain_topics.split(",") for lvl in (1, 2, 3) for r in sample(data / f"{t}/level{lvl}/{a.split}.jsonl", per_topic(t))]
    gen_rows = sample(data / f"general/retain_{a.split}.jsonl", a.general) if a.general else []

    gen = Generator(a.model, max_model_len=arch.MAX_CONTEXT)
    stop = set(arch.eos_ids(gen.tok)) | set(getattr(gen, "eos_ids", None) or ())
    suffix = gen.tok(arch.answer_suffix(), add_special_tokens=False)["input_ids"]
    room = max(512, a.max_len - 512)
    jobs = [  # (source, mode, rows, sampling)
        ("calculus", "direct", [r for v in calc.values() for r in v], Sampling(min(room, 3072))),
        ("calculus", "think", [r for v in calc.values() for r in v[:int(len(v) * a.think_frac)]], math_thinking(room)),
        ("retain_math", "direct", retain, Sampling(min(room, 2048))),
        ("general", "direct", gen_rows, general(min(room, 2048))),
    ]
    out_rows, summary = [], {"model": a.model, "args": vars(a), "sets": {}}
    for source, mode, rows, s in jobs:
        if not rows:
            continue
        prompts = [gen.chat_ids(r["instruction"], thinking=False) for r in rows]  # no thinking mode on Aya
        outs, perf = gen.generate(prompts, s)
        st = {"asked": len(rows), "truncated": 0, "too_long": 0, "rejected": 0, "kept": 0, "perf": perf, "sampling": s.as_dict()}
        for r, p, o in zip(rows, prompts, outs):
            if o["finish_reason"] != "stop":
                st["truncated"] += 1
                continue
            comp = close_turn(list(o["token_ids"]), stop, suffix)
            if len(p) + len(comp) > a.max_len:
                st["too_long"] += 1
                continue
            text = arch.clean(o["text"])
            if not comp:
                ok = False
            elif source == "general":
                _, final, _ = grade.split_thinking(text)
                ok = bool(final) and not grade.is_gibberish(final)
            else:
                ok = grade_item(r, text, thinking=False)["correct"]
            if not ok:
                st["rejected"] += 1
                continue
            out_rows.append({"id": r["id"], "source": source, "topic": r.get("topic", "general"), "level": r.get("level"),
                             "mode": mode, "prompt_ids": p, "completion_ids": comp})
            st["kept"] += 1
        summary["sets"][f"{source}_{mode}"] = st
        print(f"[selfgen] {source}/{mode}: " + json.dumps({k: v for k, v in st.items() if k not in ('perf', 'sampling')}), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    summary |= {"rows": len(out_rows), "seconds": round(time.time() - t0, 1)}
    Path(f"{out}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[selfgen] wrote {len(out_rows)} rows to {out}", flush=True)
    target_src = "calculus" if arch.MATH_TARGET == "calculus" else "retain_math"
    n_target = sum(r["source"] == target_src and r["topic"] == arch.MATH_TARGET for r in out_rows)
    print(f"[selfgen] {n_target} rows of the math target {arch.MATH_TARGET}", flush=True)
    if not n_target:
        sys.exit(f"no {arch.MATH_TARGET} (arch.MATH_TARGET) rows kept")


if __name__ == "__main__":
    main()
