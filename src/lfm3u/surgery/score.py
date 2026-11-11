"""Fast HF dev scorer for weight surgery (dev splits only; never opens a test file).

The numbers do not match scripts/08_dev_eval.py (vLLM, larger subsets, 3,072-token cap): always compare with the
unedited target scored by this module (results/surgery/target_ref/dev.json).

Teacher-forced tasks (seconds)
  nll_calc         target's own correct calculus dev solutions (selfgen_dev_v3), mean NLL per answer token   [forget]
  nll_retain_math  same for algebra / precalculus dev solutions                                             [retain]
  nll_book_forget  NLL of the true continuation of the verbmem passages (15 unseen dev + 25 train)           [forget]
  nll_book_retain  NLL of raw_retain{1,2}_dev chunks (60)                                                    [retain]
  ppl_web          perplexity on the 64 held-out FineWeb anchor docs (the last 64 of work/fineweb_anchor.jsonl,
                   as in 08_dev_eval), first 1,024 tokens
  ppl_webnarr      perplexity on 60 held-out dialogue-heavy narrative web docs (FineWeb anchor, no Harry Potter),
                   first 512 tokens: catches a patch that damages narrative prose in general
  mmlu_val         200 MMLU validation questions, 5-shot, argmax over the letter tokens (same items as 08_dev_eval)
Generation tasks (greedy)
  verbmem_unseen   15 held-out dev passages, 128 new tokens, ROUGE-L F1 + gibberish rate
  verbmem_seen     25 of the 60 train passages
  knowmem_forget   74 forget questions (train + dev), MUSE ICL prompt: fact presence, ROUGE-L recall, flags
  knowmem_retain   76 retain questions
  calc             calculus dev, n per level, chat template with thinking off: accuracy + truncated / gibberish /
                   no-answer rates and mean tokens
  retain_math      arithmetic / algebra / precalculus dev, n per topic and level
Gibberish = grade.is_gibberish (repeated word 4-grams) OR `degenerate` (repeated character 8-grams; added after
the first book session, where a jammed model emitted ":::::" and "11111" runs that the word-level flag scored as
clean). Fluent nonsense is still missed, so generations are stored (<out>/generations*/<task>.jsonl) and should be
read before a patch is called clean.
"""
from __future__ import annotations

import json
import math
import random
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SELFGEN_DEV = ROOT / "work/surgery/selfgen_dev_v3.jsonl"
SELFGEN_TRAIN = ROOT / "work/surgery/selfgen_v3.jsonl"
ANCHOR = ROOT / "work/fineweb_anchor.jsonl"
ANCHOR_VAL = 64

TEACHER = ("nll_calc", "nll_retain_math", "nll_book_forget", "nll_book_retain", "ppl_web", "ppl_webnarr")
PRESETS = {
    "fast": TEACHER,
    "book": (*TEACHER, "verbmem_unseen", "verbmem_seen", "knowmem_forget", "knowmem_retain"),
    "calc": (*TEACHER, "calc", "retain_math"),
    "full": (*TEACHER, "mmlu_val", "verbmem_unseen", "verbmem_seen", "knowmem_forget", "knowmem_retain", "calc", "retain_math"),
}
DEFAULTS = {"calc_per_level": 20, "retain_per_level": 4, "calc_max_new": 2048, "retain_max_new": 1536, "nll_items": 48,
            "seen_passages": 25, "kv_tokens": 130_000, "seed": 0}


def log(msg: str) -> None:
    print(f"[score {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean(xs, nd: int = 4):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), nd) if xs else None


def seeded(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def read_dev(rel: str) -> list[dict]:
    from lfm3u.data.splits import read_jsonl

    return read_jsonl(ROOT / "data" / rel)  # raises on test files (frozen manifest)


# ------------------------------------------------------------------ prompts

def chat_ids(tok, text: str, thinking: bool = False) -> list[int]:
    t = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
                                enable_thinking=thinking)
    return tok(t, add_special_tokens=False)["input_ids"]


def raw_ids(tok, text: str) -> list[int]:
    return [tok.bos_token_id, *tok(text, add_special_tokens=False)["input_ids"]]


def eos_ids(tok) -> list[int]:
    return sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0})


# ------------------------------------------------------------------ model calls

def generate(model, tok, prompts: list[list[int]], max_new: int, kv_tokens: int = 110_000) -> list[dict]:
    """Greedy batched generation (left padding). Batch size keeps batch * (prompt + new tokens) under kv_tokens."""
    import torch

    pad, eos = tok.pad_token_id, eos_ids(tok)
    dev = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[dict | None] = [None] * len(prompts)
    b = 0
    while b < len(order):
        bs = 1
        while b + bs < len(order) and (bs + 1) * (len(prompts[order[b + bs]]) + max_new) <= kv_tokens:
            bs += 1
        idx = order[b:b + bs]
        b += bs
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[j, L - len(prompts[i]):] = 1
        with torch.no_grad():
            g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False,
                               temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=pad,
                               eos_token_id=eos, use_cache=True)
        for j, i in enumerate(idx):
            new = g[j, L:].tolist()
            n = next((k for k, t in enumerate(new) if t in eos or t == pad), len(new))
            outs[i] = {"text": tok.decode(new[:n], skip_special_tokens=True).strip(), "n_tokens": n, "truncated": n >= max_new}
        del g
        torch.cuda.empty_cache()
    return outs


def hidden(model, ids, att):
    """Final hidden states (after the final norm) [B, L, d]."""
    return model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state


def seq_logprobs(model, tok, seqs: list[list[int]], starts: list[int], batch_tokens: int = 12_288, chunk: int = 512) -> list[list[float]]:
    """Teacher-forced log p of tokens at positions >= start (start >= 1), one list per sequence. The lm_head runs on
    fixed-size chunks of positions: full-vocabulary logits for a whole batch do not fit in 12 GB, and chunks of varying
    size fragment the CUDA allocator until it spills into shared memory (Windows), which stalls the run."""
    import torch

    pad = tok.pad_token_id
    dev = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out: list[list[float] | None] = [None] * len(seqs)
    b = 0
    while b < len(order):
        bs = 1
        while b + bs < len(order) and (bs + 1) * len(seqs[order[b + bs]]) <= batch_tokens:
            bs += 1
        idx = order[b:b + bs]
        b += bs
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        sel = torch.zeros((len(idx), L), dtype=torch.bool)   # positions whose NEXT token is scored
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            n, s = len(seqs[i]), max(starts[i], 1)
            ids[j, :n] = torch.tensor(seqs[i])
            att[j, :n] = 1
            sel[j, s - 1:n - 1] = True
        ids, att, sel = ids.to(dev), att.to(dev), sel.to(dev)
        with torch.no_grad():
            h = hidden(model, ids, att)
            H = h[sel]                                                   # [N, d], row-major: sequence by sequence
            tgt = torch.roll(ids, -1, dims=1)[sel]
            del h
            lp = torch.empty(H.shape[0], dtype=torch.float32, device=dev)
            for c in range(0, H.shape[0], chunk):
                rows = H[c:c + chunk]
                k = rows.shape[0]
                if k < chunk:                                            # keep every logits allocation the same size
                    rows = torch.cat([rows, rows.new_zeros(chunk - k, rows.shape[1])])
                lg = torch.log_softmax(model.lm_head(rows).float(), -1)
                lp[c:c + k] = lg[:k].gather(1, tgt[c:c + k, None])[:, 0]
                del lg
        flat, pos = lp.tolist(), 0
        for j, i in enumerate(idx):
            k = int(sel[j].sum())
            out[i] = flat[pos:pos + k]
            pos += k
        del H, lp, tgt
        torch.cuda.empty_cache()
    return out


def nll_summary(lps: list[list[float]]) -> dict:
    n = sum(len(x) for x in lps)
    nll = -sum(sum(x) for x in lps) / max(n, 1)
    return {"n": len(lps), "n_tokens": n, "nll": round(nll, 4), "ppl": round(math.exp(min(nll, 50)), 3)}


def char_repetition(text: str, n: int = 8, tail: int | None = None) -> float:
    """Share of repeated character n-grams (0 = none). grade.is_gibberish counts repeated WORD 4-grams and misses
    junk without spaces (":::::::", "1111111", "63:64:65:..."), which is what a jammed model emits."""
    t = (text or "").strip()
    if tail:
        t = t[-tail:]
    if len(t) < 40:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def degenerate(text: str, math: bool = False) -> bool:
    """Calibrated on the unedited target: prose continuations and QA answers have char_repetition <= 0.10 (threshold
    0.25); math solutions repeat LaTeX legitimately, so only the last 400 characters are tested (target max 0.53,
    threshold 0.70)."""
    from lfm3u import grade

    if grade.is_gibberish(text):
        return True
    return char_repetition(text, tail=400) > 0.70 if math else char_repetition(text) > 0.25


# ------------------------------------------------------------------ grading (worker processes: sympy can hang on Windows)

def _grade_one(row: dict, text: str) -> dict:
    from lfm3u.audit.mathgrade import grade_item

    return grade_item(row, text)


_FAILED = {"answer_text": None, "parsed": None, "correct": False, "think_closed": False, "refused": False,
           "abstained": False, "gibberish": False, "grade_timeout": True}


def grade_math(rows: list[dict], texts: list[str], timeout: int = 30, procs: int = 4) -> list[dict]:
    import multiprocessing as mp

    from lfm3u import grade

    res: list[dict | None] = [None] * len(rows)
    pending = list(range(len(rows)))
    while pending:
        pool = mp.Pool(min(procs, len(pending)))
        futs = {i: pool.apply_async(_grade_one, (rows[i], texts[i])) for i in pending}
        hung = 0
        for i in pending:
            try:
                res[i] = futs[i].get(timeout=timeout)
            except mp.TimeoutError:
                res[i] = dict(_FAILED) | {"gibberish": grade.is_gibberish(texts[i])}
                hung += 1
                if hung >= procs:
                    break
            except Exception as e:  # noqa: BLE001
                res[i] = dict(_FAILED) | {"grade_error": repr(e)[:200]}
        pool.terminate()
        pool.join()
        pending = [i for i in pending if res[i] is None]
    return res


# ------------------------------------------------------------------ prompt sets for directions and activation statistics

def direction_prompts(tok, target: str, n: int = 64, split: str = "train") -> tuple[list[list[int]], list[list[int]]]:
    """(forget, contrast) prompt sets for a mean-difference direction, as in scripts/11_abliterate.py but from TRAIN
    items by default, so that the dev items stay clean for scoring.
      calculus  calculus instructions vs arithmetic / algebra / precalculus instructions (chat template, thinking off)
      book      first 128 tokens of raw_forget_train chunks vs raw_retain1_train chunks"""
    if target == "calculus":
        fr = [r for lvl in (1, 2, 3) for r in read_dev(f"calculus/level{lvl}/{split}.jsonl")]
        cr = [r for t in ("arithmetic", "algebra", "precalculus") for lvl in (1, 2, 3) for r in read_dev(f"{t}/level{lvl}/{split}.jsonl")]
        seeded("abl", "f").shuffle(fr)
        seeded("abl", "c").shuffle(cr)
        return [chat_ids(tok, r["instruction"]) for r in fr[:n]], [chat_ids(tok, r["instruction"]) for r in cr[:n]]
    fc, rc = read_dev("book/raw_forget_train.jsonl"), read_dev("book/raw_retain1_train.jsonl")
    seeded("abl", "f").shuffle(fc)
    seeded("abl", "c").shuffle(rc)
    bos = tok.bos_token_id
    return [[bos, *r["input_ids"][:128]] for r in fc[:n]], [[bos, *r["input_ids"][:128]] for r in rc[:n]]


def book_chunks(tok, kind: str, n: int, seed: int = 0, max_len: int = 512) -> list[list[int]]:
    """BOS + token ids of raw book chunks (train only): kind = "forget" (raw_forget_train) or "retain" (raw_retain{1,2}_train)."""
    rows = read_dev("book/raw_forget_train.jsonl") if kind == "forget" else \
        read_dev("book/raw_retain1_train.jsonl") + read_dev("book/raw_retain2_train.jsonl")
    seeded("chunks", kind, seed).shuffle(rows)
    return [[tok.bos_token_id, *r["input_ids"][:max_len]] for r in rows[:n]]


def web_chunks(tok, n: int, seed: int = 0, max_len: int = 512) -> list[list[int]]:
    """BOS + first max_len tokens of FineWeb anchor docs, excluding the 64 held-out docs used for ppl_web."""
    docs = [json.loads(line) for line in ANCHOR.read_text(encoding="utf-8-sig").splitlines() if line.strip()][:-ANCHOR_VAL]
    seeded("web", seed).shuffle(docs)
    return [[tok.bos_token_id, *tok(d["text"][:8 * max_len], add_special_tokens=False)["input_ids"][:max_len]] for d in docs[:n]]


_HP = None
_NARR = None


def narrative_docs() -> list[str]:
    """FineWeb anchor docs (not the 64 held-out ones) that are dialogue-heavy narrative prose (quotes, speech verbs,
    third-person pronouns) and do not mention Harry Potter: 210 docs, mostly feature journalism and personal stories.
    The first 150 (seeded order) are for activation statistics, the last 60 are the held-out `ppl_webnarr` set. The book
    contrast needs them: without narrative text on the retain side, "forget-specific" directions are partly just
    "narrative prose with dialogue"."""
    global _NARR
    import re

    if _NARR is None:
        hp = re.compile(r"Harry Potter|Hogwarts|Hermione|Dumbledore|Voldemort|Rowling|Quidditch", re.I)
        out = []
        for line in ANCHOR.read_text(encoding="utf-8-sig").splitlines()[:-ANCHOR_VAL]:
            if not line.strip():
                continue
            t = json.loads(line)["text"]
            h = t[:4000]
            if len(h) < 2500 or hp.search(h):
                continue
            if (len(re.findall(r'["“”]', h)) >= 12 and len(re.findall(r"\b(said|asked|replied|whispered|shouted|muttered|smiled|laughed|nodded)\b", h)) >= 4
                    and len(re.findall(r"\b(he|she|his|her|him)\b", h, re.I)) >= 20):
                out.append(t)
        seeded("narrative").shuffle(out)
        _NARR = out
    return _NARR


def narr_chunks(tok, n: int, max_len: int = 512) -> list[list[int]]:
    """BOS + first max_len tokens of the first n (<= 150) narrative web docs (the statistics part)."""
    return [[tok.bos_token_id, *tok(d[:8 * max_len], add_special_tokens=False)["input_ids"][:max_len]] for d in narrative_docs()[:min(n, 150)]]


def math_prompts(tok, topics: tuple, n: int, split: str = "train", seed: int = 0) -> list[list[int]]:
    """Chat-template prompts (thinking off) for math instructions of the given topics, n in total, from `split`."""
    rows = [r for t in topics for lvl in (1, 2, 3) for r in read_dev(f"{t}/level{lvl}/{split}.jsonl")]
    seeded("math_prompts", topics, split, seed).shuffle(rows)
    return [chat_ids(tok, r["instruction"]) for r in rows[:n]]


# ------------------------------------------------------------------ data

def selfgen(source: str, n: int, seed: int, max_len: int = 2048) -> tuple[list[list[int]], list[int]]:
    rows = [json.loads(line) for line in SELFGEN_DEV.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source"] == source and len(r["prompt_ids"]) + len(r["completion_ids"]) <= max_len]
    rows.sort(key=lambda r: r["id"])
    seeded("selfgen", source, seed).shuffle(rows)
    rows = rows[:n]
    return [r["prompt_ids"] + r["completion_ids"] for r in rows], [len(r["prompt_ids"]) for r in rows]


def selfgen_train(source: str, n: int, seed: int = 0, max_len: int = 1024, mode: str = "direct") -> list[list[int]]:
    """The target's own TRAIN-split solutions (s3://lexsi-hackathon/unlearn/selfgen_v3.jsonl -> work/surgery/), prompt +
    completion truncated to max_len tokens. source: calculus | retain_math | general. For activation statistics."""
    rows = [json.loads(line) for line in SELFGEN_TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source"] == source and r.get("mode", mode) == mode]
    rows.sort(key=lambda r: r["id"])
    seeded("selfgen_train", source, seed).shuffle(rows)
    return [(r["prompt_ids"] + r["completion_ids"])[:max_len] for r in rows[:n]]


def verbmem_rows(which: str, cfg: dict) -> list[dict]:
    if which == "unseen":
        return read_dev("book/verbmem_forget_dev.jsonl")
    rows = read_dev("book/verbmem_forget_train.jsonl")
    seeded("verbmem_seen", cfg["seed"]).shuffle(rows)
    return rows[:cfg["seen_passages"]]


def math_rows(topics: tuple, per_level: int, seed: int) -> list[dict]:
    """Same shuffle as scripts/08_dev_eval.py, so these are the first per_level of its items."""
    items = []
    for t in topics:
        for lvl in (1, 2, 3):
            rows = read_dev(f"{t}/level{lvl}/dev.jsonl")
            random.Random(zlib.crc32(f"{t}|{lvl}|{seed}".encode())).shuffle(rows)
            items += rows[:per_level]
    return items


def anchor_docs() -> list[str]:
    docs = [json.loads(line) for line in ANCHOR.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return [d["text"] for d in docs[-ANCHOR_VAL:]]


# ------------------------------------------------------------------ tasks: each returns (summary, rows)

def t_nll_selfgen(model, tok, cfg, source):
    seqs, starts = selfgen(source, cfg["nll_items"], cfg["seed"])
    return nll_summary(seq_logprobs(model, tok, seqs, starts)), []


def t_nll_book_forget(model, tok, cfg):
    rows = verbmem_rows("unseen", cfg) + verbmem_rows("seen", cfg)
    seqs, starts = [], []
    for r in rows:
        p = raw_ids(tok, r["prompt"])
        seqs.append(p + tok(r["target"], add_special_tokens=False)["input_ids"])
        starts.append(len(p))
    lps = seq_logprobs(model, tok, seqs, starts)
    n_un = len(verbmem_rows("unseen", cfg))
    return nll_summary(lps) | {"nll_unseen": nll_summary(lps[:n_un])["nll"], "nll_seen": nll_summary(lps[n_un:])["nll"]}, []


def t_nll_book_retain(model, tok, cfg):
    chunks = read_dev("book/raw_retain1_dev.jsonl") + read_dev("book/raw_retain2_dev.jsonl")
    seeded("book_retain", cfg["seed"]).shuffle(chunks)
    seqs = [[tok.bos_token_id, *r["input_ids"]] for r in chunks[:60]]
    return nll_summary(seq_logprobs(model, tok, seqs, [1] * len(seqs))), []


def t_ppl_web(model, tok, cfg):
    seqs = [raw_ids(tok, d)[:1024] for d in anchor_docs()]
    return nll_summary(seq_logprobs(model, tok, seqs, [1] * len(seqs))), []


def t_ppl_webnarr(model, tok, cfg):
    seqs = [raw_ids(tok, d[:8 * 512])[:513] for d in narrative_docs()[150:]]
    return nll_summary(seq_logprobs(model, tok, seqs, [1] * len(seqs))), []


def t_mmlu_val(model, tok, cfg):
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    from lfm3u.audit import utility

    rows = pq.read_table(hf_hub_download("cais/mmlu", "all/validation-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    idx = sorted(random.Random(cfg["seed"]).sample(range(len(rows)), 200))
    items = [rows[i] for i in idx]
    shots = utility.mmlu_shots()
    letters = utility.letter_tokens(tok)
    allowed = torch.tensor(sorted(letters))
    prompts = [raw_ids(tok, utility.mmlu_prompt(r, shots)) for r in items]
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for p, r in zip(prompts, items):
            h = hidden(model, torch.tensor([p], device=dev), None)[0, -1]
            lg = model.lm_head(h).float()[allowed.to(dev)]
            pred = letters[int(allowed[int(lg.argmax())])]
            out.append({"subject": r["subject"], "gold": utility.LETTERS[r["answer"]], "pred": pred,
                        "correct": pred == utility.LETTERS[r["answer"]]})
    return {"n": len(out), "acc": mean(r["correct"] for r in out)}, out


def t_verbmem(model, tok, cfg, which):
    from lfm3u import grade

    src = verbmem_rows(which, cfg)
    outs = generate(model, tok, [raw_ids(tok, r["prompt"]) for r in src], 128, cfg["kv_tokens"])
    rows = [{"id": r["id"], "output": o["text"], **grade.rouge_l(o["text"], r["target"]), "gibberish": degenerate(o["text"])}
            for r, o in zip(src, outs)]
    return {"n": len(rows), "rougeL_f1": mean(r["rougeL_f1"] for r in rows), "gibberish": mean(r["gibberish"] for r in rows)}, rows


def t_knowmem(model, tok, cfg, split):
    from lfm3u import grade
    from lfm3u.audit import muse

    src = read_dev(f"book/knowmem_{split}_qa_train.jsonl") + read_dev(f"book/knowmem_{split}_qa_dev.jsonl")
    prefix = muse.icl_prefix(read_dev(f"book/knowmem_{split}_qa_icl_dev.jsonl"))
    outs = generate(model, tok, [raw_ids(tok, prefix + f"Question: {r['question']}\nAnswer:") for r in src], 32, cfg["kv_tokens"])
    rows = []
    for r, o in zip(src, outs):
        text = o["text"].split("\nQuestion")[0].split("\n\n")[0].strip()
        rows.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "output": text} | grade.grade_text(text, r["answer"])
                    | {"gibberish": degenerate(text)})
    return {"n": len(rows), "fact": mean(r["fact"] for r in rows), "rougeL_recall": mean(r["rougeL_recall"] for r in rows),
            "abstained": mean(r["abstained"] for r in rows), "refused": mean(r["refused"] for r in rows),
            "gibberish": mean(r["gibberish"] for r in rows)}, rows


def t_math(model, tok, cfg, topics, per_level, max_new):
    items = math_rows(topics, per_level, cfg["seed"])
    outs = generate(model, tok, [chat_ids(tok, r["instruction"]) for r in items], max_new, cfg["kv_tokens"])
    grades = grade_math(items, [o["text"] for o in outs])
    rows = [{k: r.get(k) for k in ("id", "topic", "level", "kind", "answer")} | {"output": o["text"], "n_tokens": o["n_tokens"],
            "truncated": o["truncated"]} | g | {"gibberish": bool(g["gibberish"]) or degenerate(o["text"], math=True)}
            for r, o, g in zip(items, outs, grades)]
    summ = {"n": len(rows), "acc": mean(r["correct"] for r in rows), "truncated": mean(r["truncated"] for r in rows),
            "gibberish": mean(r["gibberish"] for r in rows), "no_answer": mean(not r["answer_text"] for r in rows),
            "refused": mean(r["refused"] for r in rows), "abstained": mean(r["abstained"] for r in rows),
            "mean_tokens": mean((r["n_tokens"] for r in rows), 1), "max_new": max_new,
            "grade_timeouts": sum(1 for r in rows if r.get("grade_timeout"))}
    for key in sorted({(r["topic"], r["level"]) for r in rows}):
        summ[f"{key[0]}_L{key[1]}"] = mean(r["correct"] for r in rows if (r["topic"], r["level"]) == key)
    return summ, rows


RUNNERS = {
    "nll_calc": lambda m, t, c: t_nll_selfgen(m, t, c, "calculus"),
    "nll_retain_math": lambda m, t, c: t_nll_selfgen(m, t, c, "retain_math"),
    "nll_book_forget": t_nll_book_forget,
    "nll_book_retain": t_nll_book_retain,
    "ppl_web": t_ppl_web,
    "ppl_webnarr": t_ppl_webnarr,
    "mmlu_val": t_mmlu_val,
    "verbmem_unseen": lambda m, t, c: t_verbmem(m, t, c, "unseen"),
    "verbmem_seen": lambda m, t, c: t_verbmem(m, t, c, "seen"),
    "knowmem_forget": lambda m, t, c: t_knowmem(m, t, c, "forget"),
    "knowmem_retain": lambda m, t, c: t_knowmem(m, t, c, "retain"),
    "calc": lambda m, t, c: t_math(m, t, c, ("calculus",), c["calc_per_level"], c["calc_max_new"]),
    "retain_math": lambda m, t, c: t_math(m, t, c, ("arithmetic", "algebra", "precalculus"), c["retain_per_level"], c["retain_max_new"]),
}


def evaluate(model, tok, tasks="fast", cfg: dict | None = None, out_dir: Path | None = None, tag: str = "") -> dict:
    """Run the tasks (a preset name or a list); returns {"tasks": {...}, "brief": {...}}. Generations go to
    <out_dir>/generations<tag>/<task>.jsonl."""
    cfg = DEFAULTS | (cfg or {})
    names = PRESETS[tasks] if isinstance(tasks, str) else tuple(tasks)
    res, t0 = {}, time.time()
    for name in names:
        t1 = time.time()
        summ, rows = RUNNERS[name](model, tok, cfg)
        summ["seconds"] = round(time.time() - t1, 1)
        res[name] = summ
        if rows and out_dir is not None:
            p = Path(out_dir) / f"generations{tag}" / f"{name}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"{name}: {json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)})}")
    return {"tasks": res, "brief": brief(res), "cfg": cfg, "seconds": round(time.time() - t0, 1)}


BRIEF = {"nll_calc": ("nll", "nll_calc"), "nll_retain_math": ("nll", "nll_retain_math"), "nll_book_forget": ("nll", "nll_book_forget"),
         "nll_book_retain": ("nll", "nll_book_retain"), "ppl_web": ("ppl", "ppl_web"), "ppl_webnarr": ("ppl", "ppl_webnarr"), "mmlu_val": ("acc", "mmlu_val"),
         "verbmem_unseen": ("rougeL_f1", "verbmem_unseen"), "verbmem_seen": ("rougeL_f1", "verbmem_seen"),
         "knowmem_forget": ("fact", "knowmem_forget_fact"), "knowmem_retain": ("fact", "knowmem_retain_fact"),
         "calc": ("acc", "calc_acc"), "retain_math": ("acc", "retain_math_acc")}


def brief(res: dict) -> dict:
    """Flat one-line view of a result: the headline number of each task plus the degenerate-output flags."""
    out = {}
    for name, (key, label) in BRIEF.items():
        if name in res:
            out[label] = res[name].get(key)
    for name, label in (("verbmem_unseen", "verbmem_unseen_gib"), ("verbmem_seen", "verbmem_seen_gib"), ("knowmem_forget", "knowmem_forget_gib")):
        if name in res:
            out[label] = res[name].get("gibberish")
    for name in ("calc", "retain_math"):
        if name in res:
            out[f"{name}_trunc"] = res[name].get("truncated")
            out[f"{name}_gib"] = res[name].get("gibberish")
            out[f"{name}_noans"] = res[name].get("no_answer")
    return out
