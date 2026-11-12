"""Weight-level attacks with no gradient steps (docs/ERASURE_RESEARCH.md experiment 1): does rounding or noising the
weights bring a forgotten capability back? (Unlearning that moves weights by less than a quantisation step can vanish
when the model is quantised; noise can knock a fragile suppression out of its basin.)

Variants, applied to every nn.Linear weight except lm_head (tied to the embedding), simulated in bf16 so no extra
library is needed:
  none     the checkpoint as saved
  int8     per-output-channel symmetric round-to-nearest, 8 bits
  int4     symmetric round-to-nearest, 4 bits, groups of 128 input columns
  noise_s  W + s * std(W) * N(0, 1) per tensor, for each s in --noise (fixed seed)
Metrics are attack 8's (scripts/11_abliterate.py eval_calc / eval_book; TEST splits, greedy). Writes
<out>/weight_attacks.json.
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


def rtn(w, bits: int, group: int | None):
    """Symmetric round-to-nearest fake quantisation of a 2-D weight (rows = output channels)."""
    import torch

    qmax = 2 ** (bits - 1) - 1
    x = w.float()
    if group:
        rows, cols = x.shape
        pad = (-cols) % group
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        g = x.reshape(rows, -1, group)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        q = (g / s).round().clamp(-qmax - 1, qmax) * s
        return q.reshape(rows, -1)[:, :cols].to(w.dtype)
    s = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return ((x / s).round().clamp(-qmax - 1, qmax) * s).to(w.dtype)


def apply(model, variant: str, seed: int = 0) -> int:
    import torch

    g = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or name.endswith("lm_head"):
                continue
            w = mod.weight
            if variant == "int8":
                w.copy_(rtn(w, 8, None))
            elif variant == "int4":
                w.copy_(rtn(w, 4, 128))
            elif variant.startswith("noise_"):
                s = float(variant.split("_", 1)[1])
                w.add_((torch.randn(w.shape, generator=g, device=w.device, dtype=torch.float32) * (s * w.float().std())).to(w.dtype))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--domain", choices=("book", "calc"), required=True)
    ap.add_argument("--noise", default="0.01,0.02,0.05")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--attack-data", default="attack_data")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--n-calc", type=int, default=100)
    ap.add_argument("--gen-bs", type=int, default=100)
    ap.add_argument("--n-retain", type=int, default=40)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out or f"results/weight_attacks/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    data, atk = Path(a.data), Path(a.attack_data)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to(H.device()).eval()
    orig = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    if a.domain == "calc":
        rows = read_jsonl(atk / "_inputs/math_calc_test.jsonl", allow_test=True)
        H.seeded("abl", "test").shuffle(rows)  # the attack-8 / relearning 100 problems
        rows, retain = rows[:a.n_calc], read_jsonl(atk / "_inputs/math_retain_test.jsonl", allow_test=True)[:a.n_retain]
        evaluate = lambda: H.eval_calc(model, tok, rows, retain, a.max_new, a.gen_bs)  # noqa: E731
        key = "calc_acc"
    else:
        evaluate = lambda: H.eval_book(model, tok, data, atk)  # noqa: E731
        key = "verbmem_f1"
    variants = ["none", "int8", "int4"] + [f"noise_{s}" for s in a.noise.split(",") if s]
    report = {"name": a.name, "model": a.model, "domain": a.domain, "variants": {}}
    t0 = time.time()
    for v in variants:
        model.load_state_dict(orig)
        n = apply(model, v)
        report["variants"][v] = evaluate() | {"tensors": n}
        shown = {k: x for k, x in report["variants"][v].items() if not k.endswith(("_bits", "_items"))}
        H.log(f"{a.name} {v}: {json.dumps(shown)} ({time.time() - t0:.0f}s)")
        (out / "weight_attacks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max(variants[1:], key=lambda v: report["variants"][v][key] or 0)
    report |= {"best_variant": best, "seconds": round(time.time() - t0, 1)}
    (out / "weight_attacks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    H.log(f"done; none {report['variants']['none'][key]}, best {best} {report['variants'][best][key]}")


if __name__ == "__main__":
    main()
