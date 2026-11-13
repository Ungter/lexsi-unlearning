"""Relearning probe for weight surgery (paper experiment P5): does a few-step LoRA fine-tune bring the edited-away
capability back? This is the attack that undid every earlier patch (docs/RESULTS.md finding 11), made available to
the editing agent as a BUDGETED tool so that it can select for edits that survive it.

Protocol (the LoRA of scripts/12_relearn.py, on other data splits):
  adapter   peft LoRA, rank 16, alpha 32, dropout 0, on LORA_TARGETS (q/k/v/o_proj, conv in/out_proj, w1/w2/w3 of all
            24 layers), AdamW lr 2e-4, batch 4, constant schedule, grad-norm clip 1.0, bf16 autocast, loss on the
            answer tokens only
  train     calculus: N (default 20; 7/7/6 per level) of the TARGET's own direct-mode solutions of TRAIN-split problems
            (work/surgery/selfgen_v3.jsonl = s3://lexsi-hackathon/unlearn/selfgen_v3.jsonl; every id is checked
            against data/calculus/level*/train.jsonl). kind="diff" | "integral" | "defint" restricts the solutions to
            one calculus kind (the cross-subskill variant: train on derivatives, read the integral accuracy).
            book: N chunks of data/book/raw_forget_train.jsonl (BOS + first 256 tokens, LM loss).
  score     at each requested step, with the adapter active: calculus = the session's dev set (score.t_math, same items,
            token cap and grader as dev_score: accuracy, truncated, gibberish, no answer, accuracy per kind);
            book = verbatim ROUGE-L on the dev passages (15 unseen dev + 25 train) and the NLL of the true continuation.
  after     the adapter is unloaded and discarded. The weights must be bit-identical: per-tensor checksums of the raw
            bits are compared before and after, and the probe raises if anything differs.
The held-out attack trains on dev solutions and scores test problems; this probe trains on train solutions and scores
dev problems, so the two share no item. It never opens a test file (score.read_dev refuses them).

Stronger variant, gains=True: besides the LoRA, every RMSNorm gain vector (the four sandwich norms of each layer, the
q/k norms and the final norm) gets a trainable fp32 additive adapter (zero init, same optimizer). The standard LoRA
cannot adapt norm gains, so an edit that zeroes gain entries is out of its reach by construction; an all-weight
attacker is not so limited. Use it whenever an edit touches a norm gain, so that the probe is not passed on a
technicality. The adapters are forward hooks: no weight is modified.

Cost: `cost(steps)` forward calls = 1 for the training (<= 100 steps) + 1 per scored step, at least 3. A probe at one
or two steps costs 3, as in the P5 brief (25 training steps on ~1.3k-token solutions plus one or two generation passes
over the dev set are about 2-3 dev_score calls of compute); each further scored step is one more generation pass.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from lfm3u.surgery import score

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]  # scripts/12_relearn.py
CALC_KINDS = ("diff", "integral", "defint")
MAX_STEPS = 100


def log(msg: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cost(steps) -> int:
    """Forward calls charged for one probe: 1 (training) + 1 per scored step, at least 3."""
    return max(3, 1 + len({int(s) for s in steps}))


def fingerprint(model, chunk: int = 1 << 24) -> dict:
    """{name: (sum of raw bits, position-weighted sum of raw bits)} per parameter, int64 with wrap-around. Any change
    of any element changes the first sum unless it is cancelled exactly; the weighted sum catches swaps."""
    import torch

    views = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}
    out = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            v = p.detach().reshape(-1)
            bits = v.view(views[v.dtype])
            s1 = s2 = 0
            for c in range(0, bits.numel(), chunk):
                b = bits[c:c + chunk].to(torch.int64)
                w = torch.arange(c, c + b.numel(), device=b.device, dtype=torch.int64) % 65521 + 1
                s1 += int(b.sum())
                s2 += int((b * w).sum())
            out[name] = (s1 & 0xFFFFFFFFFFFFFFFF, s2 & 0xFFFFFFFFFFFFFFFF)
    return out


# ------------------------------------------------------------------ training items (TRAIN split only)

def calc_train_items(n: int = 20, kind: str = "mixed", seed: int = 0, max_len: int = 3200) -> list[dict]:
    """n of the target's own direct-mode calculus solutions of TRAIN problems, spread over the levels as 7/7/6 for 20.
    kind: "mixed" or one of CALC_KINDS (kinds from data/calculus/level*/train.jsonl by id)."""
    rows = [json.loads(line) for line in score.SELFGEN_TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source"] == "calculus" and r.get("mode", "direct") == "direct"]
    kinds = {r["id"]: r["kind"] for lvl in (1, 2, 3) for r in score.read_dev(f"calculus/level{lvl}/train.jsonl")}
    stray = [r["id"] for r in rows if r["id"] not in kinds]
    if stray:
        raise RuntimeError(f"{len(stray)} self-generated solutions are not TRAIN-split problems (e.g. {stray[0]})")
    if kind != "mixed":
        if kind not in CALC_KINDS:
            raise ValueError(f"kind must be mixed or one of {CALC_KINDS}")
        rows = [r for r in rows if kinds[r["id"]] == kind]
    rows.sort(key=lambda r: r["id"])
    score.seeded("relearn_probe", "calculus", kind, seed).shuffle(rows)
    picked, used = [], set()
    for i, lvl in enumerate((1, 2, 3)):
        quota = n // 3 + (1 if i < n % 3 else 0)
        for r in [r for r in rows if int(r["level"]) == lvl][:quota]:
            picked.append(r)
            used.add(r["id"])
    picked += [r for r in rows if r["id"] not in used][:n - len(picked)]   # a level with too few solutions of this kind
    return [{"id": r["id"], "kind": kinds[r["id"]], "level": int(r["level"]),
             "ids": (r["prompt_ids"] + r["completion_ids"])[:max_len], "start": len(r["prompt_ids"])} for r in picked[:n]]


def book_train_items(tok, n: int = 20, seed: int = 0, max_len: int = 256) -> list[dict]:
    """n chunks of the unlearning forget text (raw_forget_train): BOS + the first max_len tokens, LM loss on all."""
    chunks = score.read_dev("book/raw_forget_train.jsonl")
    chunks.sort(key=lambda r: r["id"])
    score.seeded("relearn_probe", "book", seed).shuffle(chunks)
    return [{"id": r["id"], "kind": "chunk", "ids": [tok.bos_token_id, *r["input_ids"][:max_len]], "start": 1} for r in chunks[:n]]


# ------------------------------------------------------------------ scoring with the adapter active (DEV only)

def eval_calc(model, tok, cfg: dict, out_file: Path | None) -> dict:
    summ, rows = score.t_math(model, tok, cfg, ("calculus",), cfg["calc_per_level"], cfg["calc_max_new"])
    res = {k: summ[k] for k in ("n", "acc", "truncated", "gibberish", "no_answer", "mean_tokens", "max_new", "grade_timeouts")}
    res["finished_correct"] = sum(1 for r in rows if r["correct"] and not r["truncated"])
    res["finished_wrong"] = sum(1 for r in rows if not r["correct"] and not r["truncated"])
    res["acc_by_kind"] = {k: score.mean(r["correct"] for r in rows if r["kind"] == k) for k in CALC_KINDS}
    res["n_by_kind"] = {k: sum(1 for r in rows if r["kind"] == k) for k in CALC_KINDS}
    res["trunc_by_kind"] = {k: score.mean(r["truncated"] for r in rows if r["kind"] == k) for k in CALC_KINDS}
    _dump(rows, out_file)
    return res


def eval_book(model, tok, cfg: dict, out_file: Path | None) -> dict:
    un, rows_u = score.t_verbmem(model, tok, cfg, "unseen")
    se, rows_s = score.t_verbmem(model, tok, cfg, "seen")
    nll, _ = score.t_nll_book_forget(model, tok, cfg)
    _dump([r | {"set": "unseen"} for r in rows_u] + [r | {"set": "seen"} for r in rows_s], out_file)
    return {"verbmem_unseen": un["rougeL_f1"], "verbmem_seen": se["rougeL_f1"], "gib_unseen": un["gibberish"],
            "gib_seen": se["gibberish"], "nll_book_forget": nll["nll"], "nll_unseen": nll["nll_unseen"], "nll_seen": nll["nll_seen"]}


def _dump(rows: list[dict], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ the probe

def probe(model, tok, *, target: str = "calculus", steps=(25,), n_items: int = 20, kind: str = "mixed", lr: float = 2e-4,
          batch: int = 4, rank: int = 16, seed: int = 0, cfg: dict | None = None, out_dir: Path | None = None,
          max_len: int | None = None, gains: bool = False, gain_lr: float | None = None) -> dict:
    """Train a LoRA on `model` in place (the adapter modules wrap the frozen projections), score the dev set at each
    step in `steps` (0 = before any update), unload the adapter and verify that every weight has its original bits.
    gains=True adds trainable additive adapters on all RMSNorm gains (lr `gain_lr`, default `lr`).
    Returns {"steps": {step: metrics}, "brief": {...}, "losses": [...], "train_ids": [...], ...}."""
    import torch
    from peft import LoraConfig, get_peft_model

    cfg = score.DEFAULTS | (cfg or {})
    steps = sorted({int(s) for s in steps})
    if not steps or steps[0] < 0 or steps[-1] > MAX_STEPS:
        raise ValueError(f"steps must be within 0..{MAX_STEPS}")
    total = steps[-1]
    if target == "calculus":
        items = calc_train_items(n_items, kind, seed, max_len or 3200)
    elif target == "book":
        items = book_train_items(tok, n_items, seed, max_len or 256)
    else:
        raise ValueError(f"no relearning probe for target {target}")
    if len(items) < min(n_items, batch):
        raise RuntimeError(f"only {len(items)} training items for kind {kind}")
    dev = next(model.parameters()).device
    names = [n for n, _ in model.named_parameters()]
    before = fingerprint(model)
    had_cfg = hasattr(model, "peft_config")
    t0 = time.time()
    torch.manual_seed(seed)   # LoRA init; the batch order uses rng below (as in scripts/12_relearn.py)
    pm = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=list(LORA_TARGETS),
                                          task_type="CAUSAL_LM"))
    res_steps, losses, n_trainable = {}, [], 0
    gain_hooks, gain_params, gain_info = [], [], None
    try:
        params = [p for p in pm.parameters() if p.requires_grad]
        groups = [{"params": params, "lr": lr}]
        if gains:
            for _name, mod in model.named_modules():
                if type(mod).__name__.endswith("RMSNorm"):
                    d = torch.nn.Parameter(torch.zeros(mod.weight.shape, dtype=torch.float32, device=dev))

                    def hook(m, inp, out, d=d):   # out = weight * normed(x); add d * normed(x)
                        x = inp[0].float()
                        return out + (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.variance_epsilon) * d).to(out.dtype)

                    gain_hooks.append(mod.register_forward_hook(hook))
                    gain_params.append(d)
            groups.append({"params": gain_params, "lr": gain_lr or lr})
            params = params + gain_params
        n_trainable = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
        pad = tok.pad_token_id
        rng = random.Random(seed)
        order: list[int] = []
        log(f"{target}/{kind}: {len(items)} items ({sum(len(it['ids']) for it in items)} tokens), {n_trainable:,} trainable, steps {steps}")
        for step in range(total + 1):
            if step in steps:
                model.eval()
                f = None if out_dir is None else Path(out_dir) / f"step_{step:03d}.jsonl"
                r = eval_calc(model, tok, cfg, f) if target == "calculus" else eval_book(model, tok, cfg, f)
                r["train_loss"] = score.mean(losses[-10:])
                res_steps[step] = r
                log(f"step {step}: {json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)})} ({time.time() - t0:.0f}s)")
            if step == total:
                break
            model.train()
            chosen = []
            while len(chosen) < batch:
                if not order:
                    order = list(range(len(items)))
                    rng.shuffle(order)
                chosen.append(items[order.pop()])
            L = max(len(it["ids"]) for it in chosen)
            ids = torch.full((len(chosen), L), pad, dtype=torch.long)
            att = torch.zeros((len(chosen), L), dtype=torch.long)
            lab = torch.full((len(chosen), L), -100, dtype=torch.long)
            for j, it in enumerate(chosen):
                t = torch.tensor(it["ids"])
                ids[j, :len(t)], att[j, :len(t)] = t, 1
                lab[j, it["start"]:len(t)] = t[it["start"]:]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).last_hidden_state
                tgt = lab[:, 1:].to(dev)
                keep = tgt != -100   # logits only at the labelled positions (the same loss as 12_relearn, less memory)
                logits = model.lm_head(h[:, :-1][keep])
            loss = torch.nn.functional.cross_entropy(logits.float(), tgt[keep])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            del h, logits, loss
        if gains:
            gain_info = {"n_vectors": len(gain_params), "lr": gain_lr or lr,
                         "max_abs_delta": round(max(float(d.detach().abs().max()) for d in gain_params), 5),
                         "mean_abs_delta": round(sum(float(d.detach().abs().mean()) for d in gain_params) / len(gain_params), 6)}
        del opt, params
    finally:
        for hk in gain_hooks:
            hk.remove()
        del gain_params
        pm.unload()   # put the original projections back; the adapter weights are dropped with pm
        del pm
        if not had_cfg and hasattr(model, "peft_config"):
            delattr(model, "peft_config")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    after = fingerprint(model)
    if [n for n, _ in model.named_parameters()] != names or after != before:
        changed = [n for n in names if after.get(n) != before[n]]
        raise RuntimeError(f"the relearning probe changed the model: {changed[:5] or 'parameter names differ'}")
    brief = {}
    for s, r in res_steps.items():
        if target == "calculus":
            brief[str(s)] = {"acc": r["acc"], "trunc": r["truncated"], "gib": r["gibberish"], "noans": r["no_answer"],
                             **r["acc_by_kind"], "train_loss": r["train_loss"]}
        else:
            brief[str(s)] = {k: r[k] for k in ("verbmem_unseen", "verbmem_seen", "gib_unseen", "gib_seen", "nll_book_forget", "train_loss")}
    return {"target": target, "kind": kind, "n_items": len(items), "train_ids": [it["id"] for it in items],
            "train_kinds": {k: sum(1 for it in items if it["kind"] == k) for k in sorted({it["kind"] for it in items})},
            "lora": {"rank": rank, "alpha": 2 * rank, "targets": list(LORA_TARGETS), "lr": lr, "batch": batch, "seed": seed,
                     "n_trainable": n_trainable, "gain_adapters": gain_info},
            "steps": {str(s): r for s, r in res_steps.items()}, "brief": brief, "losses": [round(x, 4) for x in losses],
            "cfg": cfg, "weights_unchanged": True, "seconds": round(time.time() - t0, 1)}
