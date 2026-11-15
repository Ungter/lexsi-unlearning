"""Forget/retain batches for SafeTune unlearning on the frozen Phase-2 data.

Items are {"ids": [...], "start": k}; labels cover positions >= k (prompt tokens masked). Batches are padded dicts of
2-D tensors (input_ids, attention_mask, labels), packed by a padded-token budget, so SafeTune's trainers (which move
2-D tensors as-is) take several sequences per step.

  book      forget = book/raw_forget_train chunks (held-out verbmem passages already cut out)
            retain = book/raw_retain{1,2}_train chunks interleaved with self-generated general chat answers
  calculus  forget = the target model's own correct calculus solutions (scripts/06_selfgen.py, source=calculus)
            retain = its own correct arithmetic/algebra/precalculus solutions interleaved with general chat answers
  arithmetic / algebra / precalculus  (depth ladder, PLAN 4.7)
            forget = its own correct solutions for that topic
            retain = its own correct solutions for the other math topics, calculus included, interleaved with general
            On the Aya branch the math target is arch.MATH_TARGET = algebra (--target algebra): forget = selfgen
            retain_math rows with topic algebra, retain = arithmetic + precalculus + calculus (= arch.MATH_RETAIN).
  retain_scope="general" (any target) drops the topic retain set and keeps only the general chat answers: the depth-ladder
  variant that does not anchor the skills built on top of the forgotten one

Tokenizer (Tiny Aya port): the frozen book chunks store input_ids from whichever tokenizer built data/ (LFM3 for the
phase2-20261109 snapshot), so when a tokenizer is passed raw_items re-tokenizes the chunk text instead. Self-generated
rows are token ids too; selfgen_items refuses files whose prompts do not end with Aya's <|START_RESPONSE|>.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from lfm3u import arch
from lfm3u.data.splits import read_jsonl

GEN_PROMPT_END = "<|START_RESPONSE|>"  # last token of every Aya generation prompt (arch.chat_ids)


def raw_items(path: Path, bos: int | None, tok=None, max_len: int | None = None) -> list[dict]:
    """[BOS] + chunk ids, labels on every chunk token. With tok, the ids come from re-tokenizing the chunk's text (the
    stored input_ids may belong to another tokenizer); max_len caps the item length including BOS."""
    head = [bos] if bos is not None else []
    out = []
    for r in read_jsonl(path):
        ids = list(tok(r["text"], add_special_tokens=False)["input_ids"]) if tok is not None and r.get("text") \
            else list(r["input_ids"])
        if max_len:
            ids = ids[:max_len - len(head)]
        if ids:
            out.append({"ids": [*head, *ids], "start": len(head), "src": path.stem})
    return out


LADDER = ("arithmetic", "algebra", "precalculus")


def math_topic(row: dict) -> str | None:
    """Math topic of a selfgen row (calculus rows have source 'calculus', the other topics source 'retain_math');
    None for general rows."""
    src = row.get("source")
    return "calculus" if src == "calculus" else row.get("topic") if src == "retain_math" else None


def topic_items(path: Path, topics: tuple, max_len: int, tok=None) -> list[dict]:
    """selfgen_items for any set of math topics (e.g. (arch.MATH_TARGET,) or arch.MATH_RETAIN)."""
    out = selfgen_items(path, "calculus", max_len, tok=tok) if "calculus" in topics else []
    rest = tuple(t for t in topics if t != "calculus")
    return out + (selfgen_items(path, "retain_math", max_len, topics=rest, tok=tok) if rest else [])


def selfgen_items(path: Path, source: str, max_len: int, topics: tuple | None = None, tok=None) -> list[dict]:
    """Rows of scripts/06_selfgen.py output; labels on the completion (answer + end tokens). With tok, rows whose
    prompt does not end with <|START_RESPONSE|> are dropped, and a file that is mostly such rows is an error."""
    gp = tok.convert_tokens_to_ids(GEN_PROMPT_END) if tok is not None else None
    max_len = min(max_len, arch.MAX_CONTEXT)
    out, seen, foreign = [], 0, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["source"] != source or (topics is not None and r.get("topic") not in topics):
            continue
        seen += 1
        if gp is not None and (not r["prompt_ids"] or r["prompt_ids"][-1] != gp):
            foreign += 1
            continue
        ids = r["prompt_ids"] + r["completion_ids"]
        if len(ids) <= max_len:
            out.append({"ids": ids, "start": len(r["prompt_ids"]), "src": f"{source}_{r.get('mode', '')}"})
    if foreign * 2 > seen > 0:
        raise ValueError(f"{path}: {foreign}/{seen} {source} rows were tokenized for another model (e.g. LFM3); "
                         "regenerate with scripts/06_selfgen.py")
    return out


def interleave(*lists: list) -> list:
    out, its = [], [iter(x) for x in lists]
    while its:
        nxt = []
        for it in its:
            v = next(it, None)
            if v is not None:
                out.append(v)
                nxt.append(it)
        its = nxt
    return out


def pack(items: list[dict], budget: int, pad: int, rng: random.Random | None = None) -> list[dict]:
    """Length-bucketed batches with batch_size * longest <= budget (at least one item per batch)."""
    import torch

    order = list(range(len(items)))
    if rng is not None:
        rng.shuffle(order)
    groups, cur, cur_max = [], [], 0
    for i in sorted(order, key=lambda i: len(items[i]["ids"])):
        n = len(items[i]["ids"])
        if cur and max(cur_max, n) * (len(cur) + 1) > budget:
            groups.append(cur)
            cur, cur_max = [], 0
        cur.append(items[i])
        cur_max = max(cur_max, n)
    if cur:
        groups.append(cur)
    if rng is not None:
        rng.shuffle(groups)
    batches = []
    for g in groups:
        L = max(len(it["ids"]) for it in g)
        ids = torch.full((len(g), L), pad, dtype=torch.long)
        att = torch.zeros((len(g), L), dtype=torch.long)
        lab = torch.full((len(g), L), -100, dtype=torch.long)
        for j, it in enumerate(g):
            t = torch.tensor(it["ids"], dtype=torch.long)
            n = len(t)
            ids[j, :n], att[j, :n] = t, 1
            s = max(it["start"], 1)
            lab[j, s:n] = t[s:]
        batches.append({"input_ids": ids, "attention_mask": att, "labels": lab})
    return batches


def build(target: str, data: Path, tok, *, selfgen: Path | None, n_forget: int, n_retain: int, budget: int,
          max_len: int = 4096, seed: int = 0, retain_scope: str = "all") -> tuple[list[dict], list[dict], dict]:
    """(forget_batches, retain_batches, stats). The retain list is cycled to at least as many batches as forget."""
    rng = random.Random(seed)
    max_len = min(max_len, arch.MAX_CONTEXT)
    bos, pad = tok.bos_token_id, tok.pad_token_id
    if pad is None:
        pad = arch.eos_ids(tok)[0]
    general = selfgen_items(selfgen, "general", max_len, tok=tok) if selfgen else []
    if target == "book":
        raw = lambda name: raw_items(data / f"book/{name}.jsonl", bos, tok, max_len)  # noqa: E731
        forget = raw("raw_forget_train")
        retain_main = raw("raw_retain1_train") + raw("raw_retain2_train")
    elif target == "calculus":
        if not selfgen:
            raise ValueError("calculus unlearning needs --selfgen")
        forget = selfgen_items(selfgen, "calculus", max_len, tok=tok)
        retain_main = selfgen_items(selfgen, "retain_math", max_len, tok=tok)
    elif target in LADDER:
        if not selfgen:
            raise ValueError(f"{target} unlearning needs --selfgen")
        forget = selfgen_items(selfgen, "retain_math", max_len, topics=(target,), tok=tok)
        others = tuple(t for t in LADDER if t != target)
        retain_main = (selfgen_items(selfgen, "retain_math", max_len, topics=others, tok=tok)
                       + selfgen_items(selfgen, "calculus", max_len, tok=tok))
    else:
        raise ValueError(target)
    if retain_scope == "general":
        retain_main = []
    elif retain_scope != "all":
        raise ValueError(retain_scope)
    for xs in (forget, retain_main, general):
        rng.shuffle(xs)
    forget = forget[:n_forget] if n_forget else forget
    retain = interleave(retain_main, general)
    retain = retain[:n_retain] if n_retain else retain
    fb = pack(forget, budget, pad, rng)
    rb = pack(retain, budget, pad, rng)
    if not fb or not rb:
        raise ValueError(f"empty batches: forget {len(fb)}, retain {len(rb)}")
    base = list(rb)
    while len(rb) < len(fb):
        rb += base
    stats = {"forget_items": len(forget), "retain_items": len(retain), "forget_batches": len(fb),
             "retain_batches": len(rb), "forget_tokens": sum(len(x["ids"]) for x in forget),
             "retain_tokens": sum(len(x["ids"]) for x in retain), "general_items_available": len(general), "retain_scope": retain_scope}
    return fb, rb, stats
