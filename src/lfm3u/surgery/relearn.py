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

P6: the sampled SUITE (`sample_configs`, `suite`). The single configuration above was what the P5 editor selected
against, and its patch then failed under every other attack configuration (docs/RESULTS.md finding 18). A suite call
draws K attack configurations from the fixed distribution SUITE (seed not chosen by the caller), trains each on a fresh
copy of the current weights, scores the dev set at the requested steps and reports per-configuration curves plus the
max and the mean accuracy over configurations at each step. Suite configurations train only on the ATTACK pool of the
train-split solutions (`calc_pools`: 89 per level = 267 of the 667); the other 400 (EDIT pool) are the only calculus
solutions the editor's statistics may use, so attack data and edit data are disjoint, as they are for the held-out
attacks. All-weight configurations follow scripts/12_relearn.py --full: fp32 master weights on the copy, bf16 autocast in
training, AdamW, gradient checkpointing, every parameter trainable, evaluation in fp32. Cost: K * cost(steps).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

from lfm3u.surgery import score

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]  # scripts/12_relearn.py
# component-restricted attacks, as LORA_GROUPS in scripts/12_relearn.py
LORA_GROUPS = {"all": LORA_TARGETS, "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "conv": ["in_proj", "out_proj"],
               "mlp": ["w1", "w2", "w3"]}
CALC_KINDS = ("diff", "integral", "defint")
MAX_STEPS = 100
ATTACK_PER_LEVEL = 89

# The P6 suite: the pre-registered distribution of attack configurations (docs/SURGERY_P6.md section 2). Do not edit
# after a run has registered it: Session.relearn_probe_suite refuses to run when the hash differs from the registered one.
SUITE = {
    "version": "p6-v1",
    "slots": "slot i gets family perm[i % 3], perm = a random order of the three families (K = 3: one of each)",
    "families": {
        "full": {"what": "all-weight fine-tuning: fp32 master weights, bf16 autocast, AdamW, gradient checkpointing, "
                         "every parameter trainable, evaluated in fp32", "lr": [1e-5, 3e-5]},
        "lora16": {"what": "LoRA r16 (alpha 32) on all projections (LORA_TARGETS)", "lr": [2e-4, 1e-3]},
        "lorax": {"what": "LoRA of another rank or restricted to one component type (alpha = 2 r)",
                  "variants": [[4, "all"], [64, "all"], [16, "attn"], [16, "conv"]], "lr": [2e-4, 1e-3]},
    },
    "within_family": "the j-th use of a family in one call takes the j-th element of a random order of its lr list (full, "
                     "lora16) or of its variant list (lorax; lr drawn uniformly)",
    "n_items": [20, 60, 133],
    "n_items_rule": "each block of three consecutive slots gets the three sizes in random order",
    "schedule": ["constant", "cosine"],
    "schedule_rule": "uniform per slot; cosine = linear warmup over the first 10 steps, then cosine decay to 0 at step 100",
    "seed": "uniform integer in [0, 10000) per slot: item subset, LoRA init and batch order",
    "fixed": {"batch": 4, "optimizer": "AdamW, weight decay 0", "clip": 1.0, "horizon": 100, "max_len": 3200, "dropout": 0.0},
    "train_data": "n_items of the 267 ATTACK-pool solutions (89 per level of the target's own direct-mode TRAIN-split "
                  "calculus solutions), quotas n//3 (+1) per level",
    "scoring": "the session's dev calculus items (20 per level = 60, 2,048-token cap), adapter / tuned copy active",
    "draw": "seed = sha256(run | forward calls used before the call | number of earlier suite calls | digest of the "
            "current weights); the audit command derives it from the SHA-256 of a checkpoint file or replays a stored draw",
}


def suite_hash() -> str:
    return hashlib.sha256(json.dumps(SUITE, sort_keys=True).encode()).hexdigest()


def log(msg: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cost(steps) -> int:
    """Forward calls charged for one probe: 1 (training) + 1 per scored step, at least 3."""
    return max(3, 1 + len({int(s) for s in steps}))


def suite_cost(k: int, steps) -> int:
    """Forward calls charged for a suite call: every configuration is charged like a single probe."""
    return int(k) * cost(steps)


def seed_from(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def weights_digest(fp: dict) -> str:
    """Short digest of a `fingerprint`: identifies the current (edited) weights in the suite's draw seed."""
    return hashlib.sha256(json.dumps(sorted((k, list(v)) for k, v in fp.items())).encode()).hexdigest()[:16]


def config_name(c: dict) -> str:
    head = f"full lr{c['lr']:g}" if c["full"] else f"lora r{c['rank']} {c['targets']} lr{c['lr']:g}"
    return f"{head} n{c['n_items']} {c['schedule']} s{c['seed']}"


def sample_configs(seed: int, k: int = 3) -> list[dict]:
    """K attack configurations from SUITE, deterministic in `seed`."""
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = random.Random(seed)
    fams = rng.sample(["full", "lora16", "lorax"], 3)
    order = {"full": rng.sample(SUITE["families"]["full"]["lr"], 2), "lora16": rng.sample(SUITE["families"]["lora16"]["lr"], 2),
             "lorax": rng.sample(SUITE["families"]["lorax"]["variants"], 4)}
    out, sizes = [], []
    for i in range(k):
        if i % 3 == 0:
            sizes = rng.sample(SUITE["n_items"], 3)
        fam, j = fams[i % 3], i // 3
        c = {"slot": i, "family": fam, "n_items": sizes[i % 3], "schedule": rng.choice(SUITE["schedule"]),
             "seed": rng.randrange(10000), "batch": SUITE["fixed"]["batch"]}
        if fam == "full":
            c |= {"full": True, "lr": order[fam][j % 2], "rank": None, "targets": None}
        elif fam == "lora16":
            c |= {"full": False, "lr": order[fam][j % 2], "rank": 16, "targets": "all"}
        else:
            r, t = order[fam][j % 4]
            c |= {"full": False, "lr": rng.choice(SUITE["families"][fam]["lr"]), "rank": r, "targets": t}
        c["warmup"] = 10 if c["schedule"] == "cosine" else 0
        c["name"] = config_name(c)
        out.append(c)
    return out


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

def calc_pools() -> dict:
    """Fixed split of the target's direct-mode TRAIN calculus solutions (667): "attack" = 89 per level (267), the only
    solutions a suite configuration trains on; "edit" = the other 400, the only ones edit statistics may be built from."""
    rows = [json.loads(line) for line in score.SELFGEN_TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source"] == "calculus" and r.get("mode", "direct") == "direct"]
    attack, edit = set(), set()
    for lvl in (1, 2, 3):
        ids = sorted(r["id"] for r in rows if int(r["level"]) == lvl)
        score.seeded("relearn_suite", "pool", lvl).shuffle(ids)
        attack |= set(ids[:ATTACK_PER_LEVEL])
        edit |= set(ids[ATTACK_PER_LEVEL:])
    return {"attack": attack, "edit": edit}


def calc_train_items(n: int = 20, kind: str = "mixed", seed: int = 0, max_len: int = 3200, pool: str | None = None) -> list[dict]:
    """n of the target's own direct-mode calculus solutions of TRAIN problems, spread over the levels as 7/7/6 for 20.
    kind: "mixed" or one of CALC_KINDS (kinds from data/calculus/level*/train.jsonl by id). pool: None = all 667 (the
    P5 probe), "attack" / "edit" = that part of `calc_pools` only (the P6 suite uses "attack")."""
    rows = [json.loads(line) for line in score.SELFGEN_TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["source"] == "calculus" and r.get("mode", "direct") == "direct"]
    kinds = {r["id"]: r["kind"] for lvl in (1, 2, 3) for r in score.read_dev(f"calculus/level{lvl}/train.jsonl")}
    stray = [r["id"] for r in rows if r["id"] not in kinds]
    if stray:
        raise RuntimeError(f"{len(stray)} self-generated solutions are not TRAIN-split problems (e.g. {stray[0]})")
    if pool is not None:
        keep = calc_pools()[pool]
        rows = [r for r in rows if r["id"] in keep]
    if kind != "mixed":
        if kind not in CALC_KINDS:
            raise ValueError(f"kind must be mixed or one of {CALC_KINDS}")
        rows = [r for r in rows if kinds[r["id"]] == kind]
    rows.sort(key=lambda r: r["id"])
    score.seeded(*(("relearn_probe", "calculus", kind, seed) if pool is None else ("relearn_suite", pool, kind, seed))).shuffle(rows)
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
          max_len: int | None = None, gains: bool = False, gain_lr: float | None = None, targets: str = "all",
          schedule: str = "constant", warmup: int = 0, horizon: int | None = None, full: bool = False,
          pool: str | None = None) -> dict:
    """Train a LoRA on `model` in place (the adapter modules wrap the frozen projections), score the dev set at each
    step in `steps` (0 = before any update), unload the adapter and verify that every weight has its original bits.
    gains=True adds trainable additive adapters on all RMSNorm gains (lr `gain_lr`, default `lr`).
    P6 options (defaults = the P5 probe): targets = a LORA_GROUPS key (all | attn | conv | mlp); schedule = constant |
    cosine (linear warmup over `warmup` steps, cosine to 0 at `horizon`, default the last scored step); pool = restrict
    the calculus training items to that part of `calc_pools`; full=True fine-tunes EVERY weight of `model` in place
    (scripts/12_relearn.py --full: the caller passes a throwaway fp32 copy, see `suite`; no adapter, no bit check).
    Returns {"steps": {step: metrics}, "brief": {...}, "losses": [...], "train_ids": [...], ...}."""
    import torch
    from peft import LoraConfig, get_peft_model

    cfg = score.DEFAULTS | (cfg or {})
    steps = sorted({int(s) for s in steps})
    if not steps or steps[0] < 0 or steps[-1] > MAX_STEPS:
        raise ValueError(f"steps must be within 0..{MAX_STEPS}")
    total = steps[-1]
    horizon = horizon or total
    if targets not in LORA_GROUPS or schedule not in ("constant", "cosine") or (full and gains):
        raise ValueError("targets must be a LORA_GROUPS key, schedule constant | cosine, and full excludes gains")
    if target == "calculus":
        items = calc_train_items(n_items, kind, seed, max_len or 3200, pool)
    elif target == "book":
        items = book_train_items(tok, n_items, seed, max_len or 256)
    else:
        raise ValueError(f"no relearning probe for target {target}")
    if len(items) < min(n_items, batch):
        raise RuntimeError(f"only {len(items)} training items for kind {kind}")
    dev = next(model.parameters()).device
    names = [n for n, _ in model.named_parameters()]
    before = None if full else fingerprint(model)
    had_cfg = hasattr(model, "peft_config")
    t0 = time.time()
    torch.manual_seed(seed)   # LoRA init; the batch order uses rng below (as in scripts/12_relearn.py)
    if full:
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError("full=True needs an fp32 copy of the model (bf16 AdamW updates at lr ~1e-5 mostly round away)")
        for p in model.parameters():
            p.requires_grad_(True)
        # checkpointing only acts in train mode, so the scoring passes still generate with the KV cache
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        pm = None
    else:
        pm = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                              target_modules=list(LORA_GROUPS[targets]), task_type="CAUSAL_LM"))
    res_steps, losses, n_trainable = {}, [], 0
    gain_hooks, gain_params, gain_info = [], [], None
    try:
        params = [p for p in (model if full else pm).parameters() if p.requires_grad]
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
        sched = None
        if schedule != "constant" or warmup:   # as scripts/12_relearn.py: linear warmup, then cosine decay to 0 at the horizon

            def lr_factor(s: int) -> float:
                if s < warmup:
                    return (s + 1) / warmup
                if schedule == "cosine":
                    return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, horizon - warmup))))
                return 1.0
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
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
            if sched is not None:
                sched.step()
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
        if pm is not None:
            pm.unload()   # put the original projections back; the adapter weights are dropped with pm
        del pm
        if not had_cfg and hasattr(model, "peft_config"):
            delattr(model, "peft_config")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if not full:   # a full fine-tune runs on a throwaway copy that is meant to change; `suite` checks the session model
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
            "lora": {"rank": None if full else rank, "alpha": None if full else 2 * rank,
                     "targets": "every parameter" if full else list(LORA_GROUPS[targets]), "lr": lr, "batch": batch, "seed": seed,
                     "n_trainable": n_trainable, "gain_adapters": gain_info, "full": full, "schedule": schedule, "warmup": warmup,
                     "horizon": horizon, "pool": pool},
            "steps": {str(s): r for s, r in res_steps.items()}, "brief": brief, "losses": [round(x, 4) for x in losses],
            "diverged": any(not math.isfinite(x) for x in losses),
            "cfg": cfg, "weights_unchanged": not full, "seconds": round(time.time() - t0, 1)}


def suite(model, tok, configs: list[dict], *, steps=(25, 100), cfg: dict | None = None, out_dir: Path | None = None,
          test: dict | None = None) -> dict:
    """Run the attack configurations (from `sample_configs`, or a stored draw) one after the other, each on a fresh copy
    of `model` (bf16 for LoRA, fp32 for all-weight fine-tuning) that is deleted afterwards; `model` itself is never
    wrapped or trained, and its per-tensor checksums are compared before and after the whole suite. Calculus only.
    test = {"n_items": .., "max_len": ..} shrinks every configuration (selftest only).
    Returns {"configs", "per_config": [{config, brief, ...}], "agg": {step: {max_acc, mean_acc, argmax, ...}}}."""
    import copy
    import gc

    import torch

    steps = sorted({int(s) for s in steps})
    names = [n for n, _ in model.named_parameters()]
    before = fingerprint(model)
    t0, per = time.time(), []
    for c in configs:
        log(f"suite config {c['slot']}: {c['name']}")
        work = copy.deepcopy(model)
        if c["full"]:
            work = work.float()
        try:
            r = probe(work, tok, target="calculus", steps=steps, n_items=(test or {}).get("n_items", c["n_items"]), lr=c["lr"],
                      batch=c["batch"], rank=c["rank"] or 16, seed=c["seed"], cfg=cfg,
                      out_dir=None if out_dir is None else Path(out_dir) / f"cfg{c['slot']}",
                      max_len=(test or {}).get("max_len", SUITE["fixed"]["max_len"]), targets=c["targets"] or "all",
                      schedule=c["schedule"], warmup=c["warmup"], horizon=SUITE["fixed"]["horizon"], full=c["full"], pool="attack")
        finally:
            del work
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        per.append({"config": c, "brief": r["brief"], "steps": r["steps"], "losses": r["losses"], "diverged": r["diverged"],
                    "train_ids": r["train_ids"], "train_kinds": r["train_kinds"], "n_trainable": r["lora"]["n_trainable"],
                    "seconds": r["seconds"]})
    after = fingerprint(model)
    if [n for n, _ in model.named_parameters()] != names or after != before:
        changed = [n for n in names if after.get(n) != before[n]]
        raise RuntimeError(f"the probe suite changed the model: {changed[:5] or 'parameter names differ'}")
    agg = {}
    for s in steps:
        accs = [p["brief"][str(s)]["acc"] for p in per]
        tr = [p["brief"][str(s)]["trunc"] for p in per]
        i = max(range(len(accs)), key=lambda j: accs[j])
        agg[str(s)] = {"max_acc": max(accs), "mean_acc": round(sum(accs) / len(accs), 4), "min_acc": min(accs),
                       "argmax": per[i]["config"]["name"], "trunc_at_max": tr[i], "mean_trunc": round(sum(tr) / len(tr), 4)}
    return {"suite": SUITE["version"], "suite_hash": suite_hash(), "k": len(configs), "steps": steps, "configs": configs,
            "per_config": per, "agg": agg, "weights_unchanged": True, "test": test, "cfg": score.DEFAULTS | (cfg or {}),
            "seconds": round(time.time() - t0, 1)}


def suite_brief(res: dict) -> dict:
    """Compact view: per configuration {step: [acc, truncated]}, then max / mean accuracy over configurations per step."""
    return {"configs": {p["config"]["name"]: {s: [b["acc"], b["trunc"]] for s, b in p["brief"].items()} for p in res["per_config"]},
            "max": {s: a["max_acc"] for s, a in res["agg"].items()}, "mean": {s: a["mean_acc"] for s, a in res["agg"].items()},
            "worst": {s: a["argmax"] for s, a in res["agg"].items()},
            "diverged": [p["config"]["name"] for p in res["per_config"] if p["diverged"]]}
