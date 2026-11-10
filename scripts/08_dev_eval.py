"""Fast dev-only evaluation for Phase-4 tuning (never reads a test split; ~5 min per checkpoint with vLLM).

Book forget
  verbmem_seen     book/verbmem_forget_train (60 passages inside the unlearning text), raw continuation, ROUGE-L F1
  verbmem_unseen   book/verbmem_forget_dev (15 held-out passages, cut from the unlearning text)
  knowmem_forget   book/knowmem_forget_qa_{train,dev} (74), MUSE ICL raw completion: ROUGE-L recall + fact presence
Book retain
  knowmem_retain   book/knowmem_retain_qa_{train,dev} (76), ICL
  retain_nll       NLL on book/raw_retain{1,2}_dev chunks (up to 100)
Calculus forget / math retain (direct mode, greedy; lfm3u.audit.mathgrade)
  calc             calculus/level{1,2,3}/dev, --calc-per-level each
  calc_think       calculus dev, --calc-think-per-level each, thinking mode (model-card math sampling)
  retain_math      arithmetic/algebra/precalculus level{1,2,3}/dev, --retain-per-level each
Utility
  mmlu_val         200 MMLU *validation* questions (the 1,000-question test subset is kept for the audit), 5-shot
  ifeval_think     100 IFEval prompts, thinking mode, prompt-level strict
  ppl_web          perplexity on the 64 held-out FineWeb anchor docs (never trained on), first 1,024 tokens

--system adds a system prompt to every chat-format task (the PLAN B0 prompt baseline; raw-completion tasks have no
system turn and are unaffected, which is itself a result). Writes <out>/dev.json and <out>/generations/*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import grade  # noqa: E402
from lfm3u.audit import muse, utility  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402
from lfm3u.infer import Generator, Sampling, math_thinking  # noqa: E402

TASKS = ("verbmem_seen", "verbmem_unseen", "knowmem_forget", "knowmem_retain", "retain_nll", "calc", "retain_math",
         "mmlu_val", "ppl_web", "ifeval_think", "calc_think")
ANCHOR_URI = "s3://lexsi-hackathon/eval/fineweb_anchor.jsonl"
ANCHOR_VAL = 64


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def flags(rows: list[dict]) -> dict:
    return {k: mean(r.get(k) for r in rows) for k in ("refused", "abstained", "gibberish", "truncated") if rows and k in rows[0]}


class Ctx:
    def __init__(self, gen: Generator, data: Path, out: Path, system: str | None, a):
        self.gen, self.data, self.out, self.system, self.a = gen, data, out, system, a
        self.rng = random.Random(a.seed)

    def rows(self, *paths: str) -> list[dict]:
        return [r for p in paths for r in read_jsonl(self.data / p)]

    def lim(self, rows: list) -> list:
        return rows[:self.a.limit] if self.a.limit else rows

    def chat(self, text: str, thinking: bool = False) -> list[int]:
        return self.gen.chat_ids(text, thinking=thinking, system=self.system)


def t_verbmem(c: Ctx, path: str, name: str):
    src = c.lim(c.rows(path))
    outs, _ = c.gen.generate([c.gen.raw_ids(r["prompt"]) for r in src], Sampling(128))
    rows = [{"id": r["id"], "output": o["text"], **grade.rouge_l(o["text"], r["target"]), "gibberish": grade.is_gibberish(o["text"])}
            for r, o in zip(src, outs)]
    write_jsonl(c.out / "generations" / f"{name}.jsonl", rows)
    return {"n": len(rows), "rougeL_f1": mean(r["rougeL_f1"] for r in rows), "gibberish": mean(r["gibberish"] for r in rows)}


def t_knowmem(c: Ctx, split: str, name: str):
    src = c.lim(c.rows(f"book/knowmem_{split}_qa_train.jsonl", f"book/knowmem_{split}_qa_dev.jsonl"))
    prefix = muse.icl_prefix(c.rows(f"book/knowmem_{split}_qa_icl_dev.jsonl"))
    outs, _ = c.gen.generate([c.gen.raw_ids(prefix + f"Question: {r['question']}\nAnswer:") for r in src],
                             Sampling(32, stop=("\nQuestion", "\n\n")))
    rows = [{"id": r["id"], "question": r["question"], "answer": r["answer"], "output": o["text"]} | grade.grade_text(o["text"], r["answer"])
            for r, o in zip(src, outs)]
    write_jsonl(c.out / "generations" / f"{name}.jsonl", rows)
    return {"n": len(rows), "rougeL_recall": mean(r["rougeL_recall"] for r in rows), "fact": mean(r["fact"] for r in rows)} | flags(rows)


def t_retain_nll(c: Ctx):
    chunks = c.rows("book/raw_retain1_dev.jsonl", "book/raw_retain2_dev.jsonl")
    c.rng.shuffle(chunks)
    chunks = c.lim(chunks[:100])
    bos = c.gen.tok.bos_token_id
    lps = c.gen.prompt_logprobs([[bos, *r["input_ids"]] for r in chunks])
    return utility.perplexity(lps)


def t_math(c: Ctx, topics: tuple, per_level: int, name: str, thinking: bool = False):
    items = []
    for t in topics:
        for lvl in (1, 2, 3):
            rows = c.rows(f"{t}/level{lvl}/dev.jsonl")
            random.Random(zlib.crc32(f"{t}|{lvl}|{c.a.seed}".encode())).shuffle(rows)  # str hash() is salted per process
            items += rows[:per_level]
    items = c.lim(items)
    s = math_thinking(c.a.think_max_tokens) if thinking else Sampling(c.a.direct_max_tokens)
    outs, _ = c.gen.generate([c.chat(r["instruction"], thinking=thinking) for r in items], s)
    rows = []
    for r, o in zip(items, outs):
        g = grade_item(r, o["text"], thinking=thinking)
        rows.append({k: r.get(k) for k in ("id", "topic", "level", "kind", "answer")} | {"output": o["text"], "n_tokens": o["n_tokens"],
                    "truncated": o["finish_reason"] == "length"} | g)
    write_jsonl(c.out / "generations" / f"{name}.jsonl", rows)
    summ = {"n": len(rows), "acc": mean(r["correct"] for r in rows), "mean_tokens": mean(r["n_tokens"] for r in rows)} | flags(rows)
    for key in sorted({(r["topic"], r["level"]) for r in rows}):
        sub = [r for r in rows if (r["topic"], r["level"]) == key]
        summ[f"{key[0]}_L{key[1]}"] = mean(r["correct"] for r in sub)
    return summ


def t_mmlu_val(c: Ctx):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = pq.read_table(hf_hub_download("cais/mmlu", "all/validation-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    idx = sorted(random.Random(c.a.seed).sample(range(len(rows)), 200))
    items = c.lim([rows[i] for i in idx])
    shots = utility.mmlu_shots()
    letters = utility.letter_tokens(c.gen.tok)
    outs, _ = c.gen.generate([c.gen.raw_ids(utility.mmlu_prompt(r, shots)) for r in items],
                             Sampling(1, allowed_token_ids=tuple(sorted(letters))))
    preds = [letters.get(o["token_ids"][0]) if o.get("token_ids") else None for o in outs]
    rows = [{"subject": r["subject"], "gold": utility.LETTERS[r["answer"]], "pred": p, "correct": p == utility.LETTERS[r["answer"]]}
            for r, p in zip(items, preds)]
    write_jsonl(c.out / "generations" / "mmlu_val.jsonl", rows)
    return {"n": len(rows), "acc": mean(r["correct"] for r in rows)}


def t_ifeval(c: Ctx):
    docs = utility.ifeval_items()
    docs = c.lim(sorted(random.Random(c.a.seed).sample(docs, 100), key=lambda d: d["key"]))
    from lfm3u.infer import general

    outs, _ = c.gen.generate([c.chat(d["prompt"], thinking=True) for d in docs], general(c.a.think_max_tokens))
    rows = []
    for d, o in zip(docs, outs):
        _, final, closed = grade.split_thinking(o["text"])
        final = final if closed else ""
        rows.append({"key": d["key"], "final": final, "truncated": o["finish_reason"] == "length",
                     "gibberish": grade.is_gibberish(final)} | utility.grade_ifeval(d, final))
    write_jsonl(c.out / "generations" / "ifeval_think.jsonl", rows)
    return utility.summarize_ifeval(rows) | flags(rows)


def t_ppl_web(c: Ctx):
    cache = Path("work/fineweb_anchor.jsonl")
    if not cache.is_file():
        from lfm3u import storage

        cache.parent.mkdir(parents=True, exist_ok=True)
        b, k = storage.split_uri(ANCHOR_URI)
        storage.client().download_file(b, k, str(cache))
    docs = [json.loads(line) for line in cache.read_text(encoding="utf-8-sig").splitlines() if line.strip()][-ANCHOR_VAL:]
    lps = c.gen.prompt_logprobs([c.gen.raw_ids(d["text"])[:1024] for d in c.lim(docs)])
    return utility.perplexity(lps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="default results/dev/<name>")
    ap.add_argument("--data", default="data")
    ap.add_argument("--system", default=None, help="system prompt for chat tasks (prompt baseline)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--calc-per-level", type=int, default=80)
    ap.add_argument("--calc-think-per-level", type=int, default=20)
    ap.add_argument("--retain-per-level", type=int, default=30)
    ap.add_argument("--direct-max-tokens", type=int, default=3072)
    ap.add_argument("--think-max-tokens", type=int, default=6144)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap items per task (smoke test)")
    a = ap.parse_args()

    out = Path(a.out or f"results/dev/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if t in a.tasks.split(",")]
    t0 = time.time()
    gen = Generator(a.model, max_model_len=a.max_model_len)
    c = Ctx(gen, Path(a.data), out, a.system, a)
    meta = {}
    mp = Path(a.model) / "lfm3u_meta.json"
    if mp.is_file():
        m = json.loads(mp.read_text(encoding="utf-8"))
        meta = {k: m.get(k) for k in ("name", "method", "target", "config", "data", "train_seconds", "base_model")}
    report = {"name": a.name, "model": a.model, "system": a.system, "unlearn": meta, "args": vars(a), "results": {}}
    runners = {
        "verbmem_seen": lambda: t_verbmem(c, "book/verbmem_forget_train.jsonl", "verbmem_seen"),
        "verbmem_unseen": lambda: t_verbmem(c, "book/verbmem_forget_dev.jsonl", "verbmem_unseen"),
        "knowmem_forget": lambda: t_knowmem(c, "forget", "knowmem_forget"),
        "knowmem_retain": lambda: t_knowmem(c, "retain", "knowmem_retain"),
        "retain_nll": lambda: t_retain_nll(c),
        "calc": lambda: t_math(c, ("calculus",), a.calc_per_level, "calc"),
        "calc_think": lambda: t_math(c, ("calculus",), a.calc_think_per_level, "calc_think", thinking=True),
        "retain_math": lambda: t_math(c, ("arithmetic", "algebra", "precalculus"), a.retain_per_level, "retain_math"),
        "mmlu_val": lambda: t_mmlu_val(c),
        "ifeval_think": lambda: t_ifeval(c),
        "ppl_web": lambda: t_ppl_web(c),
    }
    for t in tasks:
        t1 = time.time()
        res = runners[t]()
        res["seconds"] = round(time.time() - t1, 1)
        report["results"][t] = res
        report["seconds"] = round(time.time() - t0, 1)
        (out / "dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[dev] {a.name} {t}: {json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)})}", flush=True)
    print(f"[dev] {a.name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
