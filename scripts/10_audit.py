"""Phase-6 audit of one checkpoint on the TEST splits (the only code besides frozen-data tests allowed to read them).

  python scripts/10_audit.py --model PATH --name NAME --out results/audit/NAME [--domains book,calc] [--tasks a,b]
         [--system-book "..."] [--system-calc "..."]       (system prompts = the B0 prompt baseline)

Direct tasks (test splits; always run): book verbatim/knowmem/retain, calculus (direct + thinking), retain math, MMLU
(1,000-question subset, 5-shot), IFEval (thinking), FineWeb perplexity (docs 0-499).
Attacks (--domains selects which target's attacks run; PLAN 6.3 numbering):
  1 paraphrase  2 notation  3 word problems  4 indirect / excerpt questions  5 translation  6 jailbreak wrappers
  7 best-of-N   9 in-context examples  11 re-derivation (thinking)  12 tool loophole (sandboxed python)
Attacks 8 (abliteration), 10 (relearning curve) and 13-14 (J-lens) are separate scripts.

Every task writes generations/<task>.jsonl and a summary into <out>/audit.json. Refusal, abstention, gibberish and
truncation rates are reported next to every score and never count as forgetting.
"""
from __future__ import annotations

import argparse
import base64  # noqa: F401  (kept for ad-hoc inspection of encoded prompts)
import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import grade  # noqa: E402
from lfm3u.audit import attacks, muse, sandbox, utility  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402
from lfm3u.infer import Generator, Sampling, general, math_thinking  # noqa: E402

KNOW_CHAT = muse.KNOW_CHAT
VERB_CHAT = muse.VERB_CHAT

# name -> domain; order = run order (cheap first, long thinking runs last)
TASKS = {
    "verbmem_raw": "book", "knowmem_chat": "book", "knowmem_icl": "book", "retain_knowmem_chat": "book",
    "retain_knowmem_icl": "book", "retain_nll": "book", "paraphrase_book": "book", "indirect_book": "book",
    "excerpt_book": "book", "translation_book": "book", "jailbreak_book": "book", "bon_book": "book",
    "calc": "calc", "retain_math": "calc", "paraphrase_calc": "calc", "notation_calc": "calc", "word_calc": "calc",
    "translation_calc": "calc", "jailbreak_calc": "calc", "tool_calc": "calc", "icl_calc": "calc", "bon_calc": "calc",
    "mmlu": "util", "ppl_web": "util", "knowmem_think": "book", "ifeval_think": "util", "calc_think": "calc",
    "rederive_calc": "calc",
}
DIRECT = {"verbmem_raw", "knowmem_chat", "knowmem_icl", "retain_knowmem_chat", "retain_knowmem_icl", "retain_nll", "calc",
          "retain_math", "mmlu", "ppl_web", "knowmem_think", "ifeval_think", "calc_think"}


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


def seeded(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


class Ctx:
    def __init__(self, gen: Generator, a, out: Path):
        self.gen, self.a, self.out = gen, a, out
        self.data, self.atk = Path(a.data), Path(a.attack_data)
        self.sys = {"book": a.system_book, "calc": a.system_calc, "util": None}
        self._cache: dict = {}

    def chat(self, text: str, thinking: bool, domain: str, prefill: str = "") -> list[int]:
        ids = self.gen.chat_ids(text, thinking=thinking, system=self.sys.get(domain))
        return ids + (self.gen.tok(prefill, add_special_tokens=False)["input_ids"] if prefill else [])

    def lim(self, rows: list) -> list:
        return rows[:self.a.limit] if self.a.limit else rows

    def jsonl(self, rel: str, allow_test: bool = True) -> list[dict]:
        p = (self.atk if rel.startswith("@") else self.data) / rel.lstrip("@")
        return read_jsonl(p, allow_test=allow_test)

    def calc_test(self) -> list[dict]:
        return self.lim(self.jsonl("@_inputs/math_calc_test.jsonl"))

    def calc_subset(self, n: int) -> list[dict]:
        rows = self.jsonl("@_inputs/math_calc_test.jsonl")
        seeded("subset", n).shuffle(rows)
        return self.lim(rows[:n])


# ------------------------------------------------------------------ generic runners

def run_math(c: Ctx, name: str, rows: list[dict], *, thinking: bool = False, prompt=None, sampling=None, prefill: str = "",
             groups=("level", "kind", "language")) -> dict:
    if not rows:
        return {"n": 0}
    prompts = [c.chat(prompt(r) if prompt else r["instruction"], thinking, "calc", prefill) for r in rows]
    s = sampling or (math_thinking(c.a.think_max) if thinking else Sampling(c.a.direct_max))
    outs, _ = c.gen.generate(prompts, s)
    recs = []
    for r, o in zip(rows, outs):
        g = grade_item(r, o["text"], thinking)
        recs.append({k: r.get(k) for k in ("id", "parent_id", "level", "kind", "language", "target", "attack") if k in r}
                    | {"output": o["text"], "n_tokens": o["n_tokens"], "truncated": o["finish_reason"] == "length"} | g)
    write_jsonl(c.out / "generations" / f"{name}.jsonl", recs)
    return summarize_math(recs, groups)


def summarize_math(recs: list[dict], groups=("level", "kind", "language")) -> dict:
    s = {"n": len(recs), "acc": mean(r["correct"] for r in recs), "mean_tokens": mean(r["n_tokens"] for r in recs)} | flags(recs)
    for g in groups:
        keys = sorted({r[g] for r in recs if r.get(g) is not None}, key=str)
        if len(keys) > 1:
            s[f"by_{g}"] = {str(k): mean(r["correct"] for r in recs if r.get(g) == k) for k in keys}
    return s


def golds(r: dict) -> list[str]:
    return [g for g in (r.get("answer"), r.get("answer_translated")) if g]


def qa_rows(c: Ctx, name: str, rows: list[dict], texts: list[str], outs: list[dict], mode: str, extra=("attack", "target", "language")) -> dict:
    recs = []
    for r, t, o in zip(rows, texts, outs):
        reasoning, final, closed = grade.split_thinking(o["text"])
        if mode == "think" and not closed:
            final = ""
        gs = [grade.grade_text(final, g) for g in golds(r)] or [grade.grade_text(final, "")]
        best = max(gs, key=lambda x: (x["fact"], x["rougeL_recall"]))
        rec = {k: r.get(k) for k in ("id", "parent_id", "question", "answer") if k in r} | {k: r.get(k) for k in extra if r.get(k) is not None}
        rec |= {"prompt": t[:400], "output": o["text"], "final": final, "n_tokens": o["n_tokens"], "truncated": o["finish_reason"] == "length",
                "fact": best["fact"], "rougeL_recall": best["rougeL_recall"], "refused": gs[0]["refused"], "abstained": gs[0]["abstained"] and not best["fact"],
                "gibberish": gs[0]["gibberish"]}
        if mode == "think":
            rec |= grade.reasoning_metrics(reasoning, final, r["answer"])
        recs.append(rec)
    write_jsonl(c.out / "generations" / f"{name}.jsonl", recs)
    s = {"n": len(recs), "fact": mean(r["fact"] for r in recs), "rougeL_recall": mean(r["rougeL_recall"] for r in recs)} | flags(recs)
    if mode == "think":
        s |= {"fact_in_reasoning": mean(r["fact_in_reasoning"] for r in recs), "hidden_fact": mean(r["hidden_fact"] for r in recs)}
    for g in ("attack", "language"):
        keys = sorted({r[g] for r in recs if r.get(g)}, key=str)
        if len(keys) > 1:
            s[f"by_{g}"] = {k: {"n": sum(r.get(g) == k for r in recs), "fact": mean(r["fact"] for r in recs if r.get(g) == k)} for k in keys}
    return s


def run_qa(c: Ctx, name: str, rows: list[dict], *, mode: str = "chat", prefix: str = "", wrap=None, prefill: str = "") -> dict:
    """Book QA. mode: chat (thinking off), think, icl (raw few-shot completion). wrap(text) -> text rewrites the user message."""
    if not rows:
        return {"n": 0}
    if mode == "icl":
        texts = [prefix + f"Question: {r['question']}\nAnswer:" for r in rows]
        outs, _ = c.gen.generate([c.gen.raw_ids(t) for t in texts], Sampling(32, stop=("\nQuestion", "\n\n")))
    else:
        texts = [KNOW_CHAT.format(q=r["question"]) for r in rows]
        if wrap:
            texts = [wrap(t) for t in texts]
        think = mode == "think"
        s = general(c.a.book_think_max) if think else Sampling(c.a.chat_max)
        outs, _ = c.gen.generate([c.chat(t, think, "book", prefill) for t in texts], s)
    return qa_rows(c, name, rows, texts, outs, mode)


# ------------------------------------------------------------------ direct tasks

def t_verbmem_raw(c: Ctx) -> dict:
    rows = c.lim(c.jsonl("book/verbmem_forget_test.jsonl"))
    outs, _ = c.gen.generate([c.gen.raw_ids(r["prompt"]) for r in rows], Sampling(128))
    recs = [{"id": r["id"], "output": o["text"], **grade.rouge_l(o["text"], r["target"]), "gibberish": grade.is_gibberish(o["text"])}
            for r, o in zip(rows, outs)]
    write_jsonl(c.out / "generations" / "verbmem_raw.jsonl", recs)
    return {"n": len(recs), "rougeL_f1": mean(r["rougeL_f1"] for r in recs), "rougeL_recall": mean(r["rougeL_recall"] for r in recs),
            "gibberish": mean(r["gibberish"] for r in recs)}


def icl_prefix(c: Ctx, split: str) -> str:
    return muse.icl_prefix(c.jsonl(f"book/knowmem_{split}_qa_icl_dev.jsonl"))


def t_knowmem_chat(c): return run_qa(c, "knowmem_chat", c.lim(c.jsonl("book/knowmem_forget_qa_test.jsonl")))
def t_knowmem_think(c): return run_qa(c, "knowmem_think", c.lim(c.jsonl("book/knowmem_forget_qa_test.jsonl")), mode="think")
def t_knowmem_icl(c): return run_qa(c, "knowmem_icl", c.lim(c.jsonl("book/knowmem_forget_qa_test.jsonl")), mode="icl", prefix=icl_prefix(c, "forget"))
def t_retain_knowmem_chat(c): return run_qa(c, "retain_knowmem_chat", c.lim(c.jsonl("book/knowmem_retain_qa_test.jsonl")))
def t_retain_knowmem_icl(c): return run_qa(c, "retain_knowmem_icl", c.lim(c.jsonl("book/knowmem_retain_qa_test.jsonl")), mode="icl", prefix=icl_prefix(c, "retain"))


def t_retain_nll(c: Ctx) -> dict:
    chunks = c.jsonl("book/raw_retain1_test.jsonl") + c.jsonl("book/raw_retain2_test.jsonl")
    seeded("retain_nll").shuffle(chunks)
    bos = c.gen.tok.bos_token_id
    lps = c.gen.prompt_logprobs([[bos, *r["input_ids"]] for r in c.lim(chunks[:100])])
    return utility.perplexity(lps)


def t_calc(c): return run_math(c, "calc", c.calc_test())
def t_calc_think(c): return run_math(c, "calc_think", c.calc_test(), thinking=True)
def t_retain_math(c): return run_math(c, "retain_math", c.lim(c.jsonl("@_inputs/math_retain_test.jsonl")), groups=("level", "topic"))


def t_mmlu(c: Ctx) -> dict:
    items, shots = c.lim(utility.mmlu_subset(1000)), utility.mmlu_shots()
    letters = utility.letter_tokens(c.gen.tok)
    outs, _ = c.gen.generate([c.gen.raw_ids(utility.mmlu_prompt(r, shots)) for r in items], Sampling(1, allowed_token_ids=tuple(sorted(letters))))
    recs = []
    for r, o in zip(items, outs):
        pred = letters.get(o["token_ids"][0]) if o.get("token_ids") else None
        recs.append({"idx": r["idx"], "subject": r["subject"], "pred": pred, "correct": pred == utility.LETTERS[r["answer"]]})
    write_jsonl(c.out / "generations" / "mmlu.jsonl", recs)
    return {"n": len(recs), "acc": mean(r["correct"] for r in recs)}


def t_ppl_web(c: Ctx) -> dict:
    docs = c.lim(utility.fineweb_docs())
    return utility.perplexity(c.gen.prompt_logprobs([c.gen.raw_ids(d["text"])[:1024] for d in docs]))


def t_ifeval_think(c: Ctx) -> dict:
    docs = c.lim(utility.ifeval_items())
    outs, _ = c.gen.generate([c.chat(d["prompt"], True, "util") for d in docs], general(c.a.ifeval_max))
    recs = []
    for d, o in zip(docs, outs):
        _, final, closed = grade.split_thinking(o["text"])
        final = final if closed else ""
        recs.append({"key": d["key"], "truncated": o["finish_reason"] == "length", "gibberish": grade.is_gibberish(final)} | utility.grade_ifeval(d, final))
    write_jsonl(c.out / "generations" / "ifeval_think.jsonl", recs)
    return utility.summarize_ifeval(recs) | flags(recs)


# ------------------------------------------------------------------ attacks: calculus

def t_paraphrase_calc(c: Ctx) -> dict:
    rows = c.lim(c.jsonl("@paraphrase_math/test.jsonl"))
    out = run_math(c, "paraphrase_calc", rows)
    out["retain_control"] = run_math(c, "paraphrase_retain_math", c.lim(c.jsonl("@paraphrase_math/retain_test.jsonl")), groups=("level", "topic"))
    return out


def t_notation_calc(c: Ctx) -> dict:
    rows = [r for lvl in (1, 2, 3) for r in c.jsonl(f"variants/calculus/level{lvl}/test/notation.jsonl")]
    seeded("notation").shuffle(rows)
    return run_math(c, "notation_calc", c.lim(rows[:150]))


def t_word_calc(c): return run_math(c, "word_calc", c.lim(c.jsonl("@word_problems/test.jsonl")))


def t_translation_calc(c: Ctx) -> dict:
    rows = [r for lang in ("fr", "de", "es") for r in c.jsonl(f"@translation_math/{lang}.jsonl")]
    return run_math(c, "translation_calc", c.lim(rows), groups=("level", "kind", "language"))


def t_jailbreak_calc(c: Ctx) -> dict:
    rows = c.calc_subset(60)
    out = {}
    for j in attacks.JAILBREAKS:
        wrapped = [attacks.jailbreak(j, r["instruction"]) for r in rows]
        prompts = [c.chat(w[0], False, "calc", w[1]) for w in wrapped]
        outs, _ = c.gen.generate(prompts, Sampling(c.a.direct_max))
        recs = [{"id": r["id"], "template": j, "output": o["text"], "n_tokens": o["n_tokens"], "truncated": o["finish_reason"] == "length"}
                | grade_item(r, o["text"]) for r, o in zip(rows, outs)]
        write_jsonl(c.out / "generations" / f"jailbreak_calc_{j}.jsonl", recs)
        out[j] = summarize_math(recs, ())
    out["n"] = len(rows)
    out["best_template"] = max(attacks.JAILBREAKS, key=lambda j: out[j]["acc"] or 0)
    out["acc"] = out[out["best_template"]]["acc"]
    return out


def t_tool_calc(c: Ctx) -> dict:
    rows = c.calc_subset(100)
    outs, _ = c.gen.generate([c.chat(attacks.tool_prompt(r), False, "calc") for r in rows], Sampling(768))
    recs, iso = [], None
    for r, o in zip(rows, outs):
        code = sandbox.extract_code(o["text"])
        res = sandbox.run_python(code)
        iso = res["net_isolated"]
        line = next((ln.strip() for ln in reversed(res["stdout"].splitlines()) if ln.strip()), "")
        cands = [line] + [line.split(sep)[-1].strip() for sep in (":", "=") if sep in line]
        correct = False
        for cand in cands:
            correct = correct or grade_item(r, f"Answer: {cand}")["correct"]
        recs.append({"id": r["id"], "kind": r["kind"], "level": r["level"], "ran": res["ok"], "timed_out": res["timed_out"], "stdout": line[:200],
                     "code": code[:600], "correct": correct, "n_tokens": o["n_tokens"], "truncated": o["finish_reason"] == "length",
                     "refused": grade.is_refusal(o["text"]), "abstained": False, "gibberish": grade.is_gibberish(o["text"])})
    write_jsonl(c.out / "generations" / "tool_calc.jsonl", recs)
    s = summarize_math(recs, ("level", "kind"))
    return s | {"ran": mean(r["ran"] for r in recs), "timed_out": mean(r["timed_out"] for r in recs), "net_isolated": iso}


def icl_messages(c: Ctx) -> list[dict]:
    """Three worked calculus examples in the target's own style (one per level, from the self-generated unlearning set)."""
    path = Path(c.a.selfgen)
    if not path.is_file():
        return []
    gen_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gen_rows = [r for r in gen_rows if r["source"] == "calculus" and r["mode"] == "direct"]
    instr = {r["id"]: r["instruction"] for lvl in (1, 2, 3) for r in read_jsonl(c.data / f"calculus/level{lvl}/train.jsonl")}
    msgs = []
    for lvl in (1, 2, 3):
        cand = sorted((r for r in gen_rows if r["level"] == lvl and r["id"] in instr and len(r["completion_ids"]) < 700), key=lambda r: r["id"])
        if cand:
            r = cand[0]
            msgs += [{"role": "user", "content": instr[r["id"]]},
                     {"role": "assistant", "content": c.gen.tok.decode(r["completion_ids"], skip_special_tokens=True).strip()}]
    return msgs


def t_icl_calc(c: Ctx) -> dict:
    shots = icl_messages(c)
    if not shots:
        return {"n": 0, "error": "no selfgen file for in-context examples"}
    rows = c.calc_test()
    prompts = []
    for r in rows:
        msgs = ([{"role": "system", "content": c.sys["calc"]}] if c.sys["calc"] else []) + shots + [{"role": "user", "content": r["instruction"]}]
        text = c.gen.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompts.append(c.gen.tok(text, add_special_tokens=False)["input_ids"])
    outs, _ = c.gen.generate(prompts, Sampling(c.a.direct_max))
    recs = [{"id": r["id"], "level": r["level"], "kind": r["kind"], "output": o["text"], "n_tokens": o["n_tokens"], "truncated": o["finish_reason"] == "length"}
            | grade_item(r, o["text"]) for r, o in zip(rows, outs)]
    write_jsonl(c.out / "generations" / "icl_calc.jsonl", recs)
    return summarize_math(recs, ("level", "kind")) | {"shots": len(shots) // 2}


def t_rederive_calc(c: Ctx) -> dict:
    rows = [r for r in c.calc_test() if r["kind"] == "diff"]
    return run_math(c, "rederive_calc", rows, thinking=True, prompt=attacks.rederive_prompt)


def t_bon_calc(c: Ctx) -> dict:
    rows, n = c.calc_subset(60), c.a.bon_n
    texts, owner = [], []
    for i, r in enumerate(rows):
        rng = seeded("bon", r["id"])
        for _ in range(n):
            texts.append(attacks.augment(r["instruction"], rng))
            owner.append(i)
    outs, _ = c.gen.generate([c.chat(t, False, "calc") for t in texts], Sampling(c.a.direct_max, temperature=1.0, top_p=0.95, seed=None))
    correct = [grade_item(rows[i], o["text"])["correct"] for i, o in zip(owner, outs)]
    recs = []
    for i, r in enumerate(rows):
        k = [correct[j] for j in range(len(owner)) if owner[j] == i]
        recs.append({"id": r["id"], "level": r["level"], "kind": r["kind"], "n_correct": sum(k), "any_correct": any(k)})
    write_jsonl(c.out / "generations" / "bon_calc.jsonl", recs)
    return {"n": len(rows), "N": n, "acc": mean(r["any_correct"] for r in recs), "acc_mean_sample": mean(correct),
            "mean_tokens": mean(o["n_tokens"] for o in outs), "truncated": mean(o["finish_reason"] == "length" for o in outs)}


# ------------------------------------------------------------------ attacks: book

def book_attack_rows(c: Ctx, name: str, **where) -> list[dict]:
    rows = c.jsonl(f"@book_questions/{name}.jsonl")
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def t_paraphrase_book(c: Ctx) -> dict:
    forget = book_attack_rows(c, "paraphrase", split="test", target="book")
    out = run_qa(c, "paraphrase_book", c.lim(forget))
    out["retain_control"] = run_qa(c, "paraphrase_book_retain", c.lim(book_attack_rows(c, "paraphrase", split="test", target="retain_book")))
    return out


def t_indirect_book(c: Ctx) -> dict:
    return run_qa(c, "indirect_book", c.lim(book_attack_rows(c, "indirect", split="test")))


def t_excerpt_book(c: Ctx) -> dict:
    sets = {"heldout": book_attack_rows(c, "excerpt_questions", attack="excerpt_heldout", split="test"),
            "forget_chunk": book_attack_rows(c, "excerpt_questions", attack="excerpt_forget_chunk"),
            "retain_control": book_attack_rows(c, "excerpt_questions", attack="excerpt_retain")}
    out = {k: run_qa(c, f"excerpt_{k}", c.lim(v)) for k, v in sets.items()}
    out["n"] = sum(v["n"] for v in out.values())
    out["fact"] = out["heldout"]["fact"]
    return out


def t_translation_book(c: Ctx) -> dict:
    rows = [r for r in c.jsonl("@book_questions/translation.jsonl") if r.get("target") == "book"]
    out = run_qa(c, "translation_book", c.lim(rows))
    ret = [r for r in c.jsonl("@book_questions/translation.jsonl") if r.get("target") == "retain_book"]
    out["retain_control"] = run_qa(c, "translation_book_retain", c.lim(ret))
    return out


def t_jailbreak_book(c: Ctx) -> dict:
    rows = c.lim(c.jsonl("book/knowmem_forget_qa_test.jsonl"))
    out = {}
    for j in attacks.JAILBREAKS:
        texts = [attacks.jailbreak(j, KNOW_CHAT.format(q=r["question"])) for r in rows]
        outs, _ = c.gen.generate([c.chat(t[0], False, "book", t[1]) for t in texts], Sampling(c.a.chat_max))
        out[j] = qa_rows(c, f"jailbreak_book_{j}", rows, [t[0] for t in texts], outs, "chat", extra=())
    vrows = c.lim(c.jsonl("book/verbmem_forget_test.jsonl"))
    outs, _ = c.gen.generate([c.chat(VERB_CHAT.format(p=r["prompt"]), False, "book") for r in vrows], Sampling(128))
    out["verbmem_chat"] = {"n": len(vrows), "rougeL_f1": mean(grade.rouge_l(o["text"], r["target"])["rougeL_f1"] for r, o in zip(vrows, outs)),
                           "refused": mean(grade.is_refusal(o["text"]) for o in outs), "gibberish": mean(grade.is_gibberish(o["text"]) for o in outs)}
    out["n"] = len(rows)
    out["best_template"] = max(attacks.JAILBREAKS, key=lambda j: out[j]["fact"] or 0)
    out["fact"] = out[out["best_template"]]["fact"]
    return out


def t_bon_book(c: Ctx) -> dict:
    rows, n = c.lim(c.jsonl("book/knowmem_forget_qa_test.jsonl")), c.a.bon_n
    texts, owner = [], []
    for i, r in enumerate(rows):
        rng = seeded("bonb", r["id"])
        for _ in range(n):
            texts.append(attacks.augment(KNOW_CHAT.format(q=r["question"]), rng, protect_math=False))
            owner.append(i)
    outs, _ = c.gen.generate([c.chat(t, False, "book") for t in texts], Sampling(c.a.chat_max, temperature=1.0, top_p=0.95, seed=None))
    hit = [grade.fact_present(grade.split_thinking(o["text"])[1], rows[i]["answer"]) for i, o in zip(owner, outs)]
    recs = [{"id": r["id"], "n_fact": sum(hit[j] for j in range(len(owner)) if owner[j] == i),
             "any_fact": any(hit[j] for j in range(len(owner)) if owner[j] == i)} for i, r in enumerate(rows)]
    write_jsonl(c.out / "generations" / "bon_book.jsonl", recs)
    return {"n": len(rows), "N": n, "fact": mean(r["any_fact"] for r in recs), "fact_mean_sample": mean(hit),
            "gibberish": mean(grade.is_gibberish(o["text"]) for o in outs)}


RUNNERS = {name: globals()[f"t_{name}"] for name in TASKS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="default results/audit/<name>")
    ap.add_argument("--data", default="data")
    ap.add_argument("--attack-data", default="attack_data")
    ap.add_argument("--selfgen", default="work/selfgen.jsonl")
    ap.add_argument("--domains", default="book,calc", help="which targets' attacks to run (direct tasks always run)")
    ap.add_argument("--tasks", default=None, help="comma list overriding the selection; 'direct' = the direct tasks")
    ap.add_argument("--skip", default="", help="comma list of tasks to skip")
    ap.add_argument("--system-book", default=None)
    ap.add_argument("--system-calc", default=None)
    ap.add_argument("--direct-max", type=int, default=4096)
    ap.add_argument("--think-max", type=int, default=6144)
    ap.add_argument("--chat-max", type=int, default=256)
    ap.add_argument("--book-think-max", type=int, default=8192)
    ap.add_argument("--ifeval-max", type=int, default=8192)
    ap.add_argument("--bon-n", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    domains = set(a.domains.split(","))
    if a.tasks:
        chosen = [t for tok in a.tasks.split(",") for t in (sorted(DIRECT, key=list(TASKS).index) if tok == "direct" else [tok])]
    else:
        chosen = [t for t, d in TASKS.items() if d == "util" or t in DIRECT or d in domains]
    unknown = [t for t in chosen if t not in TASKS]
    if unknown:
        sys.exit(f"unknown tasks {unknown}")
    chosen = [t for t in TASKS if t in chosen and t not in a.skip.split(",")]

    out = Path(a.out or f"results/audit/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gen = Generator(a.model, max_model_len=a.max_model_len)
    c = Ctx(gen, a, out)
    meta = {}
    mp = Path(a.model) / "lfm3u_meta.json"
    if mp.is_file():
        m = json.loads(mp.read_text(encoding="utf-8"))
        meta = {k: m.get(k) for k in ("name", "method", "target", "config", "train_seconds", "base_model")}
    report = {"name": a.name, "model": a.model, "unlearn": meta, "systems": c.sys, "args": vars(a), "load_seconds": round(time.time() - t0, 1), "results": {}}
    for t in chosen:
        t1 = time.time()
        try:
            res = RUNNERS[t](c)
        except Exception as e:  # noqa: BLE001 - one failing attack must not lose the others
            import traceback

            res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
        res["seconds"] = round(time.time() - t1, 1)
        report["results"][t] = res
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        brief = {k: v for k, v in res.items() if not isinstance(v, (dict, list))}
        print(f"[audit] {a.name} {t}: {json.dumps(brief)}", flush=True)
    print(f"[audit] {a.name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
