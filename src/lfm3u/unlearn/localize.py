"""Attribution localizer (PLAN 5.1, 5.2, 5.5): which units carry the forget target.

Units are the input features of the three matrices that write to the residual stream:
  mlp   hidden unit i of layer l = input column i of model.layers[l].feed_forward.w2      (7,680 per layer, 2,048 params)
  head  attention head h of layer l = input columns h*128:(h+1)*128 of self_attn.o_proj (16 per layer, 262,144 params)
  conv  channel c of conv layer l = input column c of conv.out_proj                      (2,048 per layer, 2,048 params)

Score (attribution patching against the retain-mean activation). Per item, m = mean log p(answer tokens), and
  s(unit) = mean over items of  sum over positions and the unit's features of  (a - a_bar_retain) * dm/da,
the first-order estimate of how much m drops when the unit is replaced by its retain mean. s_forget uses forget items and
s_retain retain items (same formula, same a_bar); the forget-specific score is s_forget - lam * max(s_retain, 0) (lam = 1).
PLAN's plain s_forget - s_retain rewards units with negative retain attribution: on the real model that selection had
almost no forget attribution (sum 0.016 vs -0.375 retain) and overlapped the book selection by 23%, so negative retain
scores get no bonus (retain_relu=True; calculus selection forget sum 0.077, book overlap 4%).

Selection (`select`): greedy by forget-specific score per parameter, positive scores only, until the budget (default
0.5% of all parameters) is used. The random control (`random=True`) draws the same number of units per tensor from the
units not selected.

  python -m lfm3u.unlearn.localize --model M --target calculus --selfgen work/selfgen.jsonl --out results/localize/calculus

writes scores.pt (per-feature scores, the input to scripts/07_unlearn.py --mask), localize.json (H5 mass by component
type and layer, the default selection and its random control) and mass.png.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

KINDS = ("mlp", "head", "conv")


def log(msg: str) -> None:
    print(f"[localize {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def writer_modules(model) -> list[tuple[int, str, str, object]]:
    """[(layer, kind, module name, module)] for every residual-stream writer."""
    out = []
    for l, layer in enumerate(model.model.layers):
        out.append((l, "mlp", f"model.layers.{l}.feed_forward.w2", layer.feed_forward.w2))
        if hasattr(layer, "self_attn"):
            out.append((l, "head", f"model.layers.{l}.self_attn.o_proj", layer.self_attn.o_proj))
        else:
            out.append((l, "conv", f"model.layers.{l}.conv.out_proj", layer.conv.out_proj))
    return out


def _capture(mods, store: dict):
    def make(name):
        def hook(_m, args):
            store[name] = args[0]
        return hook
    return [m.register_forward_pre_hook(make(n)) for _, _, n, m in mods]


def retain_means(model, batches, mods, device) -> dict:
    """Mean input feature of every writer over the real tokens of the retain batches."""
    import torch

    sums = {n: torch.zeros(m.in_features, dtype=torch.float64, device=device) for _, _, n, m in mods}
    count, store = 0, {}
    handles = _capture(mods, store)
    try:
        with torch.no_grad():
            for b in batches:
                att = b["attention_mask"].to(device)
                model(input_ids=b["input_ids"].to(device), attention_mask=att)
                w = att[..., None].to(torch.float64)
                for n in sums:
                    sums[n] += (store[n].to(torch.float64) * w).sum((0, 1))
                count += int(att.sum())
    finally:
        for h in handles:
            h.remove()
    return {n: (s / max(count, 1)).float() for n, s in sums.items()}


def attribution(model, batches, mods, means: dict, device) -> tuple[dict, int]:
    """Per-feature attribution scores (mean over items) and the item count."""
    import torch
    import torch.nn.functional as F

    acc = {n: torch.zeros(m.in_features, dtype=torch.float64, device=device) for _, _, n, m in mods}
    names = list(acc)
    n_items, store = 0, {}
    handles = _capture(mods, store)
    emb = model.get_input_embeddings()
    try:
        for b in batches:
            ids, att, lab = (b[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
            # inputs that require grad make every activation part of the graph without trainable parameters
            x0 = emb(ids).detach().requires_grad_(True)
            logits = model(inputs_embeds=x0, attention_mask=att).logits
            tgt = lab[:, 1:]
            mask = tgt != -100
            lp = -F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tgt.clamp(min=0), reduction="none")
            m_item = (lp * mask).sum(1) / mask.sum(1).clamp(min=1)
            xs = [store[n] for n in names]
            grads = torch.autograd.grad(m_item.sum(), xs)
            w = att[..., None].float()
            with torch.no_grad():
                for n, x, g in zip(names, xs, grads):
                    acc[n] += (((x.detach().float() - means[n]) * g.float()) * w).sum((0, 1)).double()
            n_items += ids.shape[0]
            del logits, lp, xs, grads, x0
            store.clear()
    finally:
        for h in handles:
            h.remove()
    return {n: (a / max(n_items, 1)).float().cpu() for n, a in acc.items()}, n_items


def score(model, tok, forget_items, retain_items, *, budget_tokens: int = 4096, device="cuda") -> dict:
    """The scores.pt payload: {"meta": ..., "modules": {name: {layer, kind, d_out, forget, retain}}}."""
    from lfm3u.unlearn.data import pack

    mods = writer_modules(model)
    fb = pack(forget_items, budget_tokens, tok.pad_token_id)
    rb = pack(retain_items, budget_tokens, tok.pad_token_id)
    t0 = time.time()
    means = retain_means(model, rb, mods, device)
    log(f"retain means over {len(retain_items)} items ({time.time() - t0:.0f}s)")
    s_f, n_f = attribution(model, fb, mods, means, device)
    log(f"forget attribution over {n_f} items ({time.time() - t0:.0f}s)")
    s_r, n_r = attribution(model, rb, mods, means, device)
    log(f"retain attribution over {n_r} items ({time.time() - t0:.0f}s)")
    cfg = model.config
    meta = {"layer_types": list(cfg.layer_types), "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
            "d_model": int(cfg.hidden_size), "n_params": int(sum(p.numel() for p in model.parameters())),
            "n_forget": n_f, "n_retain": n_r, "seconds": round(time.time() - t0, 1)}
    modules = {n: {"layer": l, "kind": k, "d_out": int(m.out_features), "forget": s_f[n], "retain": s_r[n]} for l, k, n, m in mods}
    return {"meta": meta, "modules": modules}


def unit_scores(entry: dict, head_dim: int, which: str):
    v = entry[which]
    return v.view(-1, head_dim).sum(1) if entry["kind"] == "head" else v


def unit_width(entry: dict, head_dim: int) -> int:
    return head_dim if entry["kind"] == "head" else 1


def select(scores: dict, frac: float = 0.005, *, lam: float = 1.0, max_layer: int | None = None,
           kinds=KINDS, random_control: bool = False, seed: int = 0, retain_relu: bool = True) -> dict:
    """{"units": {module name: [unit idx]}, ...}. Budget = frac * all parameters; cost of a unit = d_out * width."""
    meta, mods = scores["meta"], scores["modules"]
    hd, budget = meta["head_dim"], int(frac * meta["n_params"])
    cands = []
    for n, e in mods.items():
        if e["kind"] not in kinds or (max_layer is not None and e["layer"] > max_layer):
            continue
        sr = unit_scores(e, hd, "retain")
        s = unit_scores(e, hd, "forget") - lam * (sr.clamp(min=0) if retain_relu else sr)
        cost = e["d_out"] * unit_width(e, hd)
        for i in (s > 0).nonzero().flatten().tolist():
            cands.append((float(s[i]) / cost, n, i, cost))
    cands.sort(reverse=True)
    units, used = {}, 0
    for _, n, i, cost in cands:
        if used + cost > budget:
            continue
        units.setdefault(n, []).append(i)
        used += cost
    if random_control:
        rng = random.Random(f"random-mask-{seed}")
        rand = {}
        for n, idx in units.items():
            e = mods[n]
            pool = sorted(set(range(len(unit_scores(e, hd, "forget")))) - set(idx))
            rand[n] = rng.sample(pool, min(len(idx), len(pool)))
        units = rand
    units = {n: sorted(v) for n, v in sorted(units.items(), key=lambda kv: (mods[kv[0]]["layer"], kv[0]))}
    by_kind = {k: sum(len(v) for n, v in units.items() if mods[n]["kind"] == k) for k in KINDS}
    by_layer = {}
    for n, v in units.items():
        by_layer[mods[n]["layer"]] = by_layer.get(mods[n]["layer"], 0) + len(v)
    return {"frac": frac, "budget_params": budget, "params": used, "lam": lam, "retain_relu": retain_relu, "max_layer": max_layer, "kinds": list(kinds),
            "random": random_control, "seed": seed, "n_units": sum(len(v) for v in units.values()), "by_kind": by_kind,
            "by_layer": by_layer, "units": units}


def column_masks(scores: dict, sel: dict) -> dict:
    """{parameter name: LongTensor of input columns that may change}."""
    import torch

    hd = scores["meta"]["head_dim"]
    out = {}
    for n, idx in sel["units"].items():
        w = unit_width(scores["modules"][n], hd)
        cols = [c for i in idx for c in range(i * w, (i + 1) * w)]
        out[f"{n}.weight"] = torch.tensor(cols, dtype=torch.long)
    return out


def category(entry: dict, layer_types: list[str]) -> str:
    t = layer_types[entry["layer"]]
    after = {"conv": "conv", "sliding_attention": "sliding", "full_attention": "global"}.get(t, t)
    return {"mlp": f"MLP after {after}", "head": f"heads ({after})", "conv": "conv channels"}[entry["kind"]]


def mass(scores: dict, lam: float = 1.0) -> dict:
    """H5: share of |attribution| per component type for forget and retain, plus the forget-specific mass per layer."""
    meta, mods = scores["meta"], scores["modules"]
    hd, lt = meta["head_dim"], meta["layer_types"]
    cats, layers = {}, {}
    for e in mods.values():
        c = cats.setdefault(category(e, lt), {"forget": 0.0, "retain": 0.0, "specific": 0.0, "params": 0, "units": 0})
        uf, ur = unit_scores(e, hd, "forget"), unit_scores(e, hd, "retain")
        sp = (uf - lam * ur).clamp(min=0)
        c["forget"] += float(uf.abs().sum())
        c["retain"] += float(ur.abs().sum())
        c["specific"] += float(sp.sum())
        c["params"] += e["d_out"] * len(e["forget"])
        c["units"] += len(uf)
        lay = layers.setdefault(e["layer"], {"type": lt[e["layer"]], "specific": 0.0, "forget": 0.0})
        lay["specific"] += float(sp.sum())
        lay["forget"] += float(uf.abs().sum())
    for key in ("forget", "retain", "specific"):
        tot = sum(c[key] for c in cats.values()) or 1.0
        for c in cats.values():
            c[f"{key}_share"] = round(c[key] / tot, 4)
    tot = sum(v["specific"] for v in layers.values()) or 1.0
    for v in layers.values():
        v["specific_share"] = round(v["specific"] / tot, 4)
    return {"by_type": cats, "by_layer": {str(k): layers[k] for k in sorted(layers)}}


def plot_mass(m: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = list(m["by_type"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = range(len(cats))
    for j, (key, lab) in enumerate((("forget_share", "forget"), ("retain_share", "retain"), ("specific_share", "forget - retain (+)"))):
        ax[0].bar([i + (j - 1) * 0.27 for i in x], [m["by_type"][c][key] for c in cats], 0.27, label=lab)
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("share of |attribution|")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{title}: attribution mass by component type", fontsize=10)
    colors = {"conv": "tab:gray", "sliding_attention": "tab:orange", "full_attention": "tab:blue"}
    layers = sorted(m["by_layer"], key=int)
    ax[1].bar([int(l) for l in layers], [m["by_layer"][l]["specific_share"] for l in layers],
              color=[colors.get(m["by_layer"][l]["type"], "k") for l in layers])
    ax[1].set_xlabel("layer (gray conv, orange sliding attention, blue global attention)")
    ax[1].set_ylabel("share of forget-specific mass")
    ax[1].set_title("forget-specific attribution by layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lfm3u.unlearn.data import interleave, raw_items, selfgen_items

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", choices=("book", "calculus"), required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--selfgen", default="work/selfgen.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-forget", type=int, default=400)
    ap.add_argument("--n-retain", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--frac", type=float, default=0.005, help="default selection budget, fraction of all parameters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", action="store_true", help="only redraw mass.png from an existing --out/scores.pt")
    a = ap.parse_args(argv)
    if a.replot:
        sc = torch.load(Path(a.out) / "scores.pt", weights_only=False)
        plot_mass(mass(sc), Path(a.out) / "mass.png", a.target)
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    data = Path(a.data)
    general = selfgen_items(Path(a.selfgen), "general", a.max_len) if Path(a.selfgen).is_file() else []
    if a.target == "book":
        forget = raw_items(data / "book/raw_forget_train.jsonl", tok.bos_token_id)
        main_retain = raw_items(data / "book/raw_retain1_train.jsonl", tok.bos_token_id) + \
            raw_items(data / "book/raw_retain2_train.jsonl", tok.bos_token_id)
    else:
        forget = selfgen_items(Path(a.selfgen), "calculus", a.max_len)
        main_retain = selfgen_items(Path(a.selfgen), "retain_math", a.max_len)
    forget = [x for x in forget if len(x["ids"]) <= a.max_len]
    for xs in (forget, main_retain, general):
        rng.shuffle(xs)
    retain = interleave(main_retain, general)[:a.n_retain]
    forget = forget[:a.n_forget]
    log(f"{a.target}: {len(forget)} forget, {len(retain)} retain items on {device}")
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=getattr(torch, a.dtype)).to(device).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    sc = score(model, tok, forget, retain, budget_tokens=a.batch_tokens, device=device)
    sc["meta"] |= {"target": a.target, "model": a.model, "max_len": a.max_len, "dtype": a.dtype, "seed": a.seed}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sc, out / "scores.pt")
    m = mass(sc)
    sel = select(sc, a.frac)
    rnd = select(sc, a.frac, random_control=True, seed=a.seed)
    rep = {"meta": sc["meta"], "mass": m, "selection": {k: v for k, v in sel.items() if k != "units"},
           "random": {k: v for k, v in rnd.items() if k != "units"}, "selection_units": sel["units"]}
    (out / "localize.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    try:
        plot_mass(m, out / "mass.png", a.target)
    except ImportError as e:  # the vLLM image has no matplotlib; plot later with --replot
        log(f"no plot: {e}")
    log(f"selection: {sel['n_units']} units, {sel['params']} params, by kind {sel['by_kind']}")
    log(f"type shares (forget/retain/specific): " + "; ".join(
        f"{c} {v['forget_share']:.2f}/{v['retain_share']:.2f}/{v['specific_share']:.2f}" for c, v in m["by_type"].items()))


if __name__ == "__main__":
    main()
