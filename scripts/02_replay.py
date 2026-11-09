"""Self-replay data for scripts/03_finetune.py: the base model's own answers, kept as token ids.

Mixing these into the book fine-tune keeps LFM3's chat format, thinking mode and calculus ability close to the base
model (the replay targets are what the base model already says, so they add no new knowledge).

  dolly    databricks/databricks-dolly-15k instructions (not Alpaca, which feeds the Phase-2 general-retain splits),
           direct mode and thinking mode, model-card "general" sampling
  calculus lfm3u.audit.calc_probe problems from seed 1 (prompts in the frozen seed-0 probe set are excluded),
           direct (greedy) and thinking (model-card math sampling); only correctly graded answers are kept

Every kept completion stopped on its own (no truncation) and ends with <|im_end|>; prompt + completion fits in
--max-len tokens. Writes <out> (jsonl) and <out>.summary.json.
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
from lfm3u.audit.calc_probe import generate_calculus  # noqa: E402
from lfm3u.infer import MODEL_ID, Generator, Sampling, general, math_thinking  # noqa: E402

DOLLY = ("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")


def load_dolly(seed: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DOLLY[0], DOLLY[1], repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, r in enumerate(rows):
        context = (r.get("context") or "").strip()
        r["id"] = f"dolly-{i}"
        r["user"] = r["instruction"].strip() + (f"\n\n{context}" if context else "")
    random.Random(seed).shuffle(rows)
    return rows


def calculus_problems(n_per_level: int) -> list[dict]:
    frozen = {p["prompt"] for p in generate_calculus(70, seed=0)}
    return [p for p in generate_calculus(n_per_level, seed=1) if p["prompt"] not in frozen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default="work/replay.jsonl")
    ap.add_argument("--n-chat", type=int, default=1200, help="dolly prompts answered in direct mode")
    ap.add_argument("--n-chat-think", type=int, default=300, help="further dolly prompts answered with thinking")
    ap.add_argument("--calc-per-level", type=int, default=80, help="calculus problems per level (both modes)")
    ap.add_argument("--max-len", type=int, default=8192, help="max prompt + completion tokens")
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()

    t0 = time.time()
    gen = Generator(a.model, max_model_len=a.max_model_len)
    dolly, chat = load_dolly(a.seed), []
    for r in dolly:
        ids = gen.chat_ids(r["user"], thinking=False)
        if len(ids) <= a.max_prompt:
            chat.append(r)
        if len(chat) == a.n_chat + a.n_chat_think:
            break
    calc = calculus_problems(a.calc_per_level)
    room = lambda limit: max(256, min(limit, a.max_len - a.max_prompt))  # noqa: E731
    jobs = [  # (source, mode, items, user text, sampling)
        ("dolly", "direct", chat[:a.n_chat], lambda r: r["user"], general(room(2048))),
        ("dolly", "think", chat[a.n_chat:], lambda r: r["user"], general(room(4096))),
        ("calculus", "direct", calc, lambda p: p["prompt"], Sampling(room(4096))),
        ("calculus", "think", calc, lambda p: p["prompt"], math_thinking(room(6144))),
    ]
    rows, summary = [], {"model": a.model, "args": vars(a), "sets": {}}
    example = None
    for source, mode, items, user, s in jobs:
        think = mode == "think"
        prompts = [gen.chat_ids(user(it), thinking=think) for it in items]
        outs, perf = gen.generate(prompts, s)
        stats = {"asked": len(items), "truncated": 0, "too_long": 0, "rejected": 0, "kept": 0, "tokens": 0,
                 "sampling": s.as_dict(), "perf": perf}
        for it, p, o in zip(items, prompts, outs):
            if o["finish_reason"] != "stop":
                stats["truncated"] += 1
                continue
            comp = list(o["token_ids"])
            if not comp or comp[-1] not in gen.eos_ids:
                comp.append(gen.im_end_id)
            if len(p) + len(comp) > a.max_len:
                stats["too_long"] += 1
                continue
            _, final, closed = grade.split_thinking(o["text"])
            if source == "calculus":
                ok = grade.grade_math(it, o["text"], thinking=think)["correct"]
            else:
                ok = bool(final) and (closed or not think) and not grade.is_gibberish(final)
            if not ok:
                stats["rejected"] += 1
                continue
            rows.append({"id": it["id"], "source": source, "mode": mode, "prompt_ids": p, "completion_ids": comp})
            stats["kept"] += 1
            stats["tokens"] += len(p) + len(comp)
            if example is None:
                example = gen.tok.decode(p + comp, skip_special_tokens=False)
        summary["sets"][f"{source}_{mode}"] = stats
        print(f"[replay] {source}/{mode}: {json.dumps({k: v for k, v in stats.items() if k not in ('sampling', 'perf')})}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary |= {"rows": len(rows), "tokens": sum(len(r["prompt_ids"]) + len(r["completion_ids"]) for r in rows),
                "seconds": round(time.time() - t0, 1)}
    Path(f"{out}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[replay] wrote {len(rows)} rows, {summary['tokens']} tokens to {out}", flush=True)
    print("[replay] example:\n" + (example or "")[:1500], flush=True)
    if not rows:
        sys.exit("no replay rows kept")


if __name__ == "__main__":
    main()
