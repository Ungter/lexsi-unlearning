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
  retain_scope="general" (any target) drops the topic retain set and keeps only the general chat answers: the depth-ladder
  variant that does not anchor the skills built on top of the forgotten one
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from lfm3u.data.splits import read_jsonl


def raw_items(path: Path, bos: int) -> list[dict]:
    return [{"ids": [bos, *r["input_ids"]], "start": 1, "src": path.stem} for r in read_jsonl(path)]


LADDER = ("arithmetic", "algebra", "precalculus")


def selfgen_items(path: Path, source: str, max_len: int, topics: tuple | None = None) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids = r["prompt_ids"] + r["completion_ids"]
        if r["source"] == source and len(ids) <= max_len and (topics is None or r.get("topic") in topics):
            out.append({"ids": ids, "start": len(r["prompt_ids"]), "src": f"{source}_{r.get('mode', '')}"})
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
    bos, pad = tok.bos_token_id, tok.pad_token_id
    general = selfgen_items(selfgen, "general", max_len) if selfgen else []
    if target == "book":
        forget = raw_items(data / "book/raw_forget_train.jsonl", bos)
        retain_main = raw_items(data / "book/raw_retain1_train.jsonl", bos) + raw_items(data / "book/raw_retain2_train.jsonl", bos)
    elif target == "calculus":
        if not selfgen:
            raise ValueError("calculus unlearning needs --selfgen")
        forget = selfgen_items(selfgen, "calculus", max_len)
        retain_main = selfgen_items(selfgen, "retain_math", max_len)
    elif target in LADDER:
        if not selfgen:
            raise ValueError(f"{target} unlearning needs --selfgen")
        forget = selfgen_items(selfgen, "retain_math", max_len, topics=(target,))
        others = tuple(t for t in LADDER if t != target)
        retain_main = selfgen_items(selfgen, "retain_math", max_len, topics=others) + selfgen_items(selfgen, "calculus", max_len)
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
