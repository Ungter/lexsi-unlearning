"""Attack 8b (DECISIONS TODO 2): training-free recovery by direction ablation at the EDITED layers only.

Attack 8 (11_abliterate.py) projects one direction out after the embedding and after every decoder layer. That also
removes the skill itself (target calculus 0.71 -> 0.07), so it cannot show that a jam is hiding the skill. Here the
direction is projected out only where the unlearning method wrote (cf. Lucki et al. 2024 on RMU), with no gradient steps:

  directions  per (site, layer): normalised mean(forget) - mean(contrast), computed ON THE UNLEARNED MODEL from the same
              train/dev prompts as attack 8 (calc: calculus vs other-math dev instructions; book: forget vs retain chunks)
  sites       resid  decoder-layer output (residual stream)
              ffn    MLP write after the sandwich norm (post_ffn_norm output, the vector added to the stream)
              w2     MLP output before the sandwich norm: removing the direction there lets post_ffn_norm rescale what is
                     left, so a jam that dominates the w2 output is removed without shrinking the rest of the write
              op     operator (attention / conv) write after its sandwich norm (masked RMU also edits conv/o_proj columns)
  modes       proj   h <- h - (h.r) r
              clamp  h <- h - relu(h.r - b) r, b = mean(contrast).r  (keeps the component at its retain-level value)
Variants: none; resid at the last edited layer only; each --sites entry with every --layers layer at once (proj, and
clamp for --clamp-sites); calc: resid/w2 with SOLUTION-token directions (the target's own dev solutions, calculus vs
retain math, teacher-forced through the attacked model, completion tokens only); a random-direction control at resid.
PRIMARY variant (declared before any result): resid at every edited layer, prompt directions. The rest are exploratory.
Optional --ref (the reference target, never used by the attack itself): an "oracle" variant that ablates
mean_attacked(forget) - mean_ref(forget) (if the oracle recovers and the attack does not, direction-finding failed; if
neither does, the jam is not a single direction), plus diagnostics: oracle cosines, cosine with the target's own
forget-contrast direction, and for each edited w2 the rank profile of W2 - W2_ref and the cosine of its top left
singular vector (the true write direction of a rank-1 patch) with the attack direction.
Metrics are those of attack 8 (11_abliterate.eval_calc / eval_book, test splits, greedy). Writes <out>/local_abl.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u.data.splits import read_jsonl  # noqa: E402


def load_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("abl", ROOT / "scripts" / "11_abliterate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = load_helpers()
mean, seeded, log, chat_ids, device = H.mean, H.seeded, H.log, H.chat_ids, H.device
SITES = ("resid", "ffn", "w2", "op")


def site_module(model, site: str, layer: int):
    L = model.model.layers[layer]
    return {"resid": L, "ffn": L.post_ffn_norm, "w2": L.feed_forward.w2, "op": L.post_operator_norm}[site]


def site_means(model, tok, prompts: list[list[int]], keys: list[tuple[str, int]], bs: int = 16, starts: list[int] | None = None) -> dict:
    """Mean fp32 activation over real tokens (from position starts[i] on, if given) at every (site, layer) in keys."""
    import torch

    pad, dev = tok.pad_token_id, next(model.parameters()).device
    cur, sums, count = {}, {}, 0

    def mk(key):
        def hook(_m, _i, out):
            cur[key] = out[0] if isinstance(out, tuple) else out
        return hook

    hooks = [site_module(model, s, l).register_forward_hook(mk((s, l))) for s, l in keys]
    try:
        for b in range(0, len(prompts), bs):
            chunk = prompts[b:b + bs]
            L = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            sel = torch.zeros((len(chunk), L), dtype=torch.bool)
            for j, p in enumerate(chunk):
                ids[j, :len(p)], att[j, :len(p)] = torch.tensor(p), 1
                sel[j, (starts[b + j] if starts else 0):len(p)] = True
            with torch.no_grad():
                model.model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False)
            m = sel.to(dev)
            for k in keys:
                s = cur[k][m].float().sum(0)
                sums[k] = s if k not in sums else sums[k] + s
            count += int(m.sum())
    finally:
        for h in hooks:
            h.remove()
    return {k: v / count for k, v in sums.items()}


def ablate(model, specs: list[tuple[str, int, object, float, str]]):
    """specs: (site, layer, unit direction r, retain-level offset b, mode). Returns hook handles."""
    import torch

    dev = next(model.parameters()).device

    def mk(r, b, mode):
        r = r.to(dev, torch.float32)

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h32 = h.float()
            c = h32 @ r
            c = c if mode == "proj" else torch.relu(c - b)
            h2 = (h32 - c.unsqueeze(-1) * r).to(h.dtype)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2
        return hook

    return [site_module(model, s, l).register_forward_hook(mk(r, b, mode)) for s, l, r, b, mode in specs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--domain", choices=("book", "calc"), required=True)
    ap.add_argument("--layers", required=True, help="edited layers, e.g. 2,3,4")
    ap.add_argument("--sites", default="resid,ffn,w2")
    ap.add_argument("--clamp-sites", default="w2", help="sites that also get a clamp variant")
    ap.add_argument("--ref", default=None, help="reference (target) model: oracle direction variant + diagnostics")
    ap.add_argument("--ref-label", default="oracle",
                    help="suffix of the reference-direction variants; 'base' = the PUBLIC base model as reference, which is a "
                         "realistic attack (the attacker has the unlearned model and the open base weights), not an oracle")
    ap.add_argument("--ref-only", action="store_true", help="run only none + the reference-direction variants")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--attack-data", default="attack_data")
    ap.add_argument("--selfgen-dev", default="work/selfgen_dev.jsonl", help="calc: solution-token directions")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--n-calc", type=int, default=100)
    ap.add_argument("--n-retain", type=int, default=40)
    ap.add_argument("--gen-bs", type=int, default=100)
    ap.add_argument("--prompt-seed", type=int, default=0, help="resample the direction prompts (0 = the attack-8 sample)")
    ap.add_argument("--calc-per-kind", type=int, default=0,
                    help="calc eval: N test problems per kind from data/calculus (the 243-problem set of the cross-subskill "
                         "runs) instead of the attack-8 100")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out or f"results/local_abl/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    data, atk = Path(a.data), Path(a.attack_data)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    load = lambda p: AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True, dtype=torch.bfloat16).to(device()).eval()  # noqa: E731
    model = load(a.model)
    n = a.n_prompts
    sol = None  # (forget seqs, forget starts, contrast seqs, contrast starts): the TARGET's own dev solutions, teacher-forced

    if a.prompt_seed and a.domain != "calc":
        raise SystemExit("--prompt-seed is implemented for calc only")
    if a.domain == "calc":  # same prompt sources and seeds as attack 8
        fr = [r for lvl in (1, 2, 3) for r in read_jsonl(data / f"calculus/level{lvl}/dev.jsonl")]
        cr = [r for t in ("arithmetic", "algebra", "precalculus") for lvl in (1, 2, 3) for r in read_jsonl(data / f"{t}/level{lvl}/dev.jsonl")]
        ps = [a.prompt_seed] if a.prompt_seed else []
        seeded("abl", "f", *ps).shuffle(fr)
        seeded("abl", "c", *ps).shuffle(cr)
        forget_p, contrast_p = [chat_ids(tok, r["instruction"]) for r in fr[:n]], [chat_ids(tok, r["instruction"]) for r in cr[:n]]
        rows = read_jsonl(atk / "_inputs/math_calc_test.jsonl", allow_test=True)
        seeded("abl", "test").shuffle(rows)
        rows = rows[:a.n_calc]
        if a.calc_per_kind:  # same selection as 12_relearn.py --calc-per-kind
            rows = []
            for k in ("diff", "integral", "defint"):
                for lvl in (1, 2, 3):
                    pool = [r for r in read_jsonl(data / f"calculus/level{lvl}/test.jsonl", allow_test=True) if r["kind"] == k]
                    seeded("relearn-eval", k, lvl).shuffle(pool)
                    rows += pool[:a.calc_per_kind // 3]
        retain = read_jsonl(atk / "_inputs/math_retain_test.jsonl", allow_test=True)[:a.n_retain]
        evaluate = lambda: H.eval_calc(model, tok, rows, retain, a.max_new, a.gen_bs)  # noqa: E731
        key = "calc_acc"
        if Path(a.selfgen_dev).exists():
            sg = [json.loads(x) for x in Path(a.selfgen_dev).read_text(encoding="utf-8").splitlines() if x.strip()]
            part = lambda src: [r for r in sg if r["source"] == src and r["mode"] == "direct"][:n]  # noqa: E731
            f, c = part("calculus"), part("retain_math")
            seq = lambda rs: [(r["prompt_ids"] + r["completion_ids"])[:1024] for r in rs]  # noqa: E731
            sol = (seq(f), [len(r["prompt_ids"]) for r in f], seq(c), [len(r["prompt_ids"]) for r in c])
    else:
        bos = tok.bos_token_id
        fc = read_jsonl(data / "book/raw_forget_train.jsonl")
        rc = read_jsonl(data / "book/raw_retain1_train.jsonl")
        seeded("abl", "f").shuffle(fc)
        seeded("abl", "c").shuffle(rc)
        forget_p, contrast_p = [[bos, *r["input_ids"][:128]] for r in fc[:n]], [[bos, *r["input_ids"][:128]] for r in rc[:n]]
        evaluate = lambda: H.eval_book(model, tok, data, atk)  # noqa: E731
        key = "verbmem_f1"

    layers = [int(x) for x in a.layers.split(",")]
    sites = [s for s in a.sites.split(",") if s]
    assert all(s in SITES for s in sites), sites
    keys = sorted({(s, l) for s in sites for l in layers} | {("resid", layers[-1])})
    t0 = time.time()
    unit = lambda v: (v / v.norm()).cpu()  # noqa: E731
    cos = lambda u, v: round(float(u.cpu() @ v.cpu() / (u.norm() * v.norm())), 3)  # noqa: E731
    mf, mc = site_means(model, tok, forget_p, keys), site_means(model, tok, contrast_p, keys)
    dirs = {("p", *k): (unit(mf[k] - mc[k]), float(mc[k].cpu() @ unit(mf[k] - mc[k]))) for k in keys}
    diag = {f"{s}@{l}": {"diff_norm": round(float((mf[(s, l)] - mc[(s, l)]).norm()), 3),
                         "forget_norm": round(float(mf[(s, l)].norm()), 3), "contrast_norm": round(float(mc[(s, l)].norm()), 3),
                         "forget_proj": round(float(mf[(s, l)].cpu() @ dirs[("p", s, l)][0]), 3),
                         "contrast_proj": round(dirs[("p", s, l)][1], 3)} for s, l in keys}
    if sol:
        sf, sc = site_means(model, tok, sol[0], keys, 4, sol[1]), site_means(model, tok, sol[2], keys, 4, sol[3])
        for k in keys:
            dirs[("s", *k)] = (unit(sf[k] - sc[k]), float(sc[k].cpu() @ unit(sf[k] - sc[k])))
            diag[f"{k[0]}@{k[1]}"] |= {"sol_diff_norm": round(float((sf[k] - sc[k]).norm()), 3), "sol_vs_prompt_cos": cos(sf[k] - sc[k], mf[k] - mc[k])}
    if a.ref:
        ref_path = a.ref
        if a.ref.startswith("s3://"):  # on the Vast box only the attacked model is fetched by the job script
            from lfm3u import storage

            ref_path = "models/_ref"
            if not (Path(ref_path) / "config.json").exists():
                storage.download_prefix(a.ref.rstrip("/") + "/", ref_path)
        ref = load(ref_path)
        rf, rc_ = site_means(ref, tok, forget_p, keys), site_means(ref, tok, contrast_p, keys)
        for k in keys:
            o = mf[k] - rf[k]
            dirs[("o", *k)] = (unit(o), float(mc[k].cpu() @ unit(o)))
            diag[f"{k[0]}@{k[1]}"] |= {"oracle_norm": round(float(o.norm()), 3), "oracle_cos": cos(o, mf[k] - mc[k]),
                                        "ref_dir_cos": cos(rf[k] - rc_[k], mf[k] - mc[k])}
        if sol:
            rs = site_means(ref, tok, sol[0], keys, 4, sol[1])
            for k in keys:
                o = sf[k] - rs[k]
                diag[f"{k[0]}@{k[1]}"] |= {"sol_oracle_rel": round(float(o.norm() / sf[k].norm()), 3), "sol_oracle_cos": cos(o, sf[k] - sc[k])}
        for l in layers:  # the actual weight change at w2: is it rank-1, and does the attack find its write direction?
            W, W0 = model.model.layers[l].feed_forward.w2.weight.detach().float(), ref.model.layers[l].feed_forward.w2.weight.detach().float()
            dW = W - W0
            if float(dW.norm()) == 0:
                diag.setdefault(f"w2@{l}", {})["dW2_rel"] = 0.0
                continue
            U, S, _ = torch.svd_lowrank(dW, q=6)
            d = diag.setdefault(f"w2@{l}", {})
            d |= {"dW2_rel": round(float(dW.norm() / W0.norm()), 4), "dW2_s1_s2": round(float(S[0] / S[1]), 2),
                  "dW2_top_cos_p": abs(cos(U[:, 0], dirs[("p", "w2", l)][0])) if ("p", "w2", l) in dirs else None}
            if ("s", "w2", l) in dirs:
                d["dW2_top_cos_s"] = abs(cos(U[:, 0], dirs[("s", "w2", l)][0]))
        del ref
        torch.cuda.empty_cache()
    log(f"{a.name} directions ({time.time() - t0:.0f}s): {json.dumps(diag)}")

    g = torch.Generator().manual_seed(0)
    for l in layers:  # random-direction control at the primary site
        v = torch.randn(model.config.hidden_size, generator=g)
        dirs[("r", "resid", l)] = (v / v.norm(), float(mc[("resid", l)].cpu() @ (v / v.norm())) if ("resid", l) in mc else 0.0)
    primary = f"resid@{a.layers}"
    variants = {"none": [], f"resid@{layers[-1]}": [("p", "resid", layers[-1], "proj")]}
    for s in sites:
        variants[f"{s}@{a.layers}"] = [("p", s, l, "proj") for l in layers]
        if s in a.clamp_sites.split(","):
            variants[f"{s}@{a.layers}:clamp"] = [("p", s, l, "clamp") for l in layers]
    if sol:
        for s in ("resid", "w2"):
            if s in sites:
                variants[f"{s}@{a.layers}:sol"] = [("s", s, l, "proj") for l in layers]
    if a.ref:
        variants[f"resid@{a.layers}:{a.ref_label}"] = [("o", "resid", l, "proj") for l in layers]
        for s in ("w2", "op"):
            if s in sites:
                variants[f"{s}@{a.layers}:{a.ref_label}"] = [("o", s, l, "proj") for l in layers]
    variants[f"resid@{a.layers}:random"] = [("r", "resid", l, "proj") for l in layers]
    if a.ref_only:
        variants = {k: v for k, v in variants.items() if k == "none" or k.endswith(f":{a.ref_label}")}
    report = {"name": a.name, "model": a.model, "domain": a.domain, "layers": layers, "sites": sites, "n_prompts": n,
              "ref": a.ref, "ref_label": a.ref_label, "primary_variant": primary if primary in variants else None, "directions": diag,
              "variants": {}}
    for name, spec in variants.items():
        hooks = ablate(model, [(s, l, *dirs[(src, s, l)], mode) for src, s, l, mode in spec])
        try:
            report["variants"][name] = evaluate()
        finally:
            for h in hooks:
                h.remove()
        log(f"{a.name} {name}: {json.dumps({k: v for k, v in report['variants'][name].items() if not k.endswith(('_bits', '_items'))})} "
            f"({time.time() - t0:.0f}s)")
        (out / "local_abl.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max((k for k in report["variants"] if k != "none"), key=lambda k: report["variants"][k][key] or 0)
    report |= {"best_variant": best, "best": report["variants"][best], "seconds": round(time.time() - t0, 1)}
    (out / "local_abl.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done; primary {primary}: {key} {g if (g := report['variants'].get(primary, {}).get(key)) is not None else '-'}; best {best}: {report['best'][key]}")


if __name__ == "__main__":
    main()
