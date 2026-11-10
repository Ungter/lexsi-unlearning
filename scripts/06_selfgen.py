"""Self-generated training targets for Phase-4 unlearning: the TARGET model's own answers (token ids).

  calculus     calculus/level{1,2,3}/train prompts; direct mode (greedy) for all, thinking mode (model-card math
               sampling) for the first --think-frac of each level; only correctly graded, untruncated answers kept
  retain_math  arithmetic/algebra/precalculus level{1,2,3}/train, direct mode; correct answers only
  general      general/retain_train (cleaned Alpaca) instructions, direct mode, model-card general sampling;
               untruncated, non-empty, non-gibberish answers

Forgetting the model's own reasoning traces is the faithful calculus target, and retaining its own style avoids the
style shift that cross-entropy on terse "Answer: <expr>" labels caused in the SafeTune smoke test. Train/dev splits
only (--split dev writes the dev-split version used by the relearning and in-context attacks). Writes <out> (jsonl: id, source, topic, level, mode, prompt_ids, completion_ids) and <out>.summary.json.
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

from lfm3u import grade  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402
from lfm3u.infer import Generator, Sampling, general, math_thinking  # noqa: E402

RETAIN_TOPICS = ("arithmetic", "algebra", "precalculus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train", choices=("train", "dev"), help="dev = relearning/ICL attack data (PLAN 6.3 #9, #10)")
    ap.add_argument("--retain-topics", default=",".join(RETAIN_TOPICS))
    ap.add_argument("--out", default="work/selfgen.jsonl")
    ap.add_argument("--calc-per-level", type=int, default=300)
    ap.add_argument("--think-frac", type=float, default=0.33)
    ap.add_argument("--retain-per-level", type=int, default=100, help="per topic and level")
    ap.add_argument("--general", type=int, default=800)
    ap.add_argument("--max-len", type=int, default=4096, help="max prompt + completion tokens kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap every set (smoke test)")
    a = ap.parse_args()

    t0 = time.time()
    data = Path(a.data)
    rng = random.Random(a.seed)
    cap = lambda n: min(n, a.limit) if a.limit else n  # noqa: E731

    def sample(path: Path, n: int) -> list[dict]:
        rows = read_jsonl(path)
        rng.shuffle(rows)
        return rows[:cap(n)]

    calc = {lvl: sample(data / f"calculus/level{lvl}/{a.split}.jsonl", a.calc_per_level) for lvl in (1, 2, 3)}
    retain = [r for t in a.retain_topics.split(",") for lvl in (1, 2, 3) for r in sample(data / f"{t}/level{lvl}/{a.split}.jsonl", a.retain_per_level)]
    gen_rows = sample(data / f"general/retain_{a.split}.jsonl", a.general) if a.general else []

    gen = Generator(a.model, max_model_len=max(a.max_len, 8192))
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
        think = mode == "think"
        prompts = [gen.chat_ids(r["instruction"], thinking=think) for r in rows]
        outs, perf = gen.generate(prompts, s)
        st = {"asked": len(rows), "truncated": 0, "too_long": 0, "rejected": 0, "kept": 0, "perf": perf, "sampling": s.as_dict()}
        for r, p, o in zip(rows, prompts, outs):
            if o["finish_reason"] != "stop":
                st["truncated"] += 1
                continue
            comp = list(o["token_ids"])
            if not comp or comp[-1] not in gen.eos_ids:
                comp.append(gen.im_end_id)
            if len(p) + len(comp) > a.max_len:
                st["too_long"] += 1
                continue
            if source == "general":
                _, final, closed = grade.split_thinking(o["text"])
                ok = bool(final) and not grade.is_gibberish(final)
            else:
                ok = grade_item(r, o["text"], thinking=think)["correct"]
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
    if not any(r["source"] == "calculus" for r in out_rows):
        sys.exit("no calculus rows kept")


if __name__ == "__main__":
    main()
