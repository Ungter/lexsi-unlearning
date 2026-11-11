"""Phase-4 unlearning driver: one method x target x config on the reference target model, via SafeTune.

  ga        unlearn.GradientAscentTrainer (grad_ascent: loss = -CE(forget)); full weights
  graddiff  unlearn.GradDiffTrainer (-CE(forget) + CE(retain)); full weights
  gakl      unlearn.GradientAscentTrainer(forget_loss="KL") (-CE(forget) + KL(model || frozen copy | retain))
  npo       unlearn.NPOTrainer variant npo_KL, LoRA on LFM3's projections (merged after training)
  npo_full  safetune.unlearn.npo_unlearn variant npo_KL on ALL weights (fp32, no LoRA; frozen fp32 reference copy)
  rmu       safetune.unlearn.rmu_unlearn, w2 of layers L-2..L, activations of layer L steered to a random direction
  ablate    zero the input columns of the --mask units (no training)
  npo_sam   npo_full with SAM on the forget loss (lfm3u.unlearn.sam; Fan et al. 2025), perturbation radius --rho

--mask SCORES (lfm3u.unlearn.localize output) restricts npo_full and rmu to the input columns of the top units
(--mask-frac of all parameters, greedy by forget-specific attribution per parameter; rmu only uses layers <= --layer);
--mask-random swaps in the same-size random control (same tensors, same unit count per tensor).

Data: lfm3u.unlearn.data.build (frozen Phase-2 train splits; calculus uses scripts/06_selfgen.py output). Batches
pack several sequences per step (--batch-tokens). SafeTune trains in fp32 (upcast); TF32 matmuls and gradient
checkpointing (not for RMU) keep it fast enough. Saves a bf16 checkpoint with lfm3u_meta.json and the run log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import ckpt  # noqa: E402
from lfm3u.unlearn.data import build  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]
METHODS = ("ga", "graddiff", "gakl", "npo", "npo_full", "npo_sam", "rmu", "ablate")
MASKABLE = ("npo_full", "rmu", "ablate")


def build_mask(a, model):
    """(column masks, selection summary) from --mask, or (None, None)."""
    if not a.mask:
        return None, None
    import torch

    from lfm3u.unlearn import localize

    scores = torch.load(a.mask, weights_only=False)
    max_layer = a.layer if a.method == "rmu" else a.mask_max_layer
    sel = localize.select(scores, a.mask_frac, lam=a.mask_lambda, max_layer=max_layer, kinds=tuple(a.mask_kinds.split(",")),
                          random_control=a.mask_random, seed=a.seed)
    summary = {k: v for k, v in sel.items() if k != "units"} | {"scores": a.mask, "scores_meta": scores["meta"]}
    return localize.column_masks(scores, sel), summary | {"units": sel["units"]}


def log(msg: str) -> None:
    print(f"[unlearn {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(method: str, model, a, forget, retain, masks=None):
    """Returns (model, mask report or None)."""
    import torch

    from safetune.runner import unlearn

    from lfm3u.unlearn import masked

    mid = a.model
    if method == "ablate":
        return model, masked.ablate(model, masks)
    if method == "ga":
        t = unlearn.GradientAscentTrainer(model, model_id=mid, forget_loss="grad_ascent", lr=a.lr, epochs=a.epochs,
                                          max_steps=a.steps, forget_clip=None)
        return t.unlearn(forget=forget, retain=retain), None
    if method == "gakl":
        t = unlearn.GradientAscentTrainer(model, model_id=mid, forget_loss="KL", lr=a.lr, epochs=a.epochs,
                                          max_steps=a.steps, forget_clip=None)
        return t.unlearn(forget=forget, retain=retain), None
    if method == "graddiff":
        t = unlearn.GradDiffTrainer(model, model_id=mid, lr=a.lr, epochs=a.epochs, max_steps=a.steps)
        return t.unlearn(forget=forget, retain=retain), None
    if method == "npo":
        import safetune

        safetune.configure(lora_target_modules=LORA_TARGETS)
        t = unlearn.NPOTrainer(model, model_id=mid, variant="npo_KL", beta=a.beta, num_epochs=a.epochs, lr=a.lr)
        return t.unlearn(forget=forget, retain=retain), None
    if method == "npo_full":
        import copy

        import safetune.unlearn as U

        model = model.float()
        dev = next(model.parameters()).device
        move = lambda bs: [{k: v.to(dev) for k, v in b.items()} for b in bs]  # noqa: E731
        reference = copy.deepcopy(model).eval()
        if hasattr(reference, "gradient_checkpointing_disable"):
            reference.gradient_checkpointing_disable()
        mk = masked.Masked(model, masks) if masks else None
        if mk:
            mk.freeze()
        else:
            for p in model.parameters():
                p.requires_grad_(True)
        # masked: no weight decay (it would shrink whole tensors; finalize() restores them either way)
        cfg = U.NPOConfig(variant="npo_KL", beta=a.beta, num_epochs=a.epochs, lr=a.lr, max_steps=a.steps or None,
                          **({"weight_decay": 0.0} if mk else {}))
        U.npo_unlearn(model, forget_batches=move(forget), retain_batches=move(retain), reference=reference, config=cfg)
        return model, (mk.finalize() if mk else None)
    if method == "npo_sam":
        import copy

        from lfm3u.unlearn.sam import sam_npo_unlearn

        model = model.float()
        dev = next(model.parameters()).device
        move = lambda bs: [{k: v.to(dev) for k, v in b.items()} for b in bs]  # noqa: E731
        reference = copy.deepcopy(model).eval()
        if hasattr(reference, "gradient_checkpointing_disable"):
            reference.gradient_checkpointing_disable()
        for p in model.parameters():
            p.requires_grad_(True)
        sam_npo_unlearn(model, move(forget), move(retain), reference, rho=a.rho, beta=a.beta, lr=a.lr, epochs=a.epochs,
                        max_steps=a.steps or None)
        return model, None
    if method == "rmu":
        import copy

        import safetune.unlearn as U

        model = model.float()
        dev = next(model.parameters()).device
        move = lambda bs: [{k: v.to(dev) for k, v in b.items()} for b in bs]  # noqa: E731
        cfg = U.RMUConfig(layer_id=a.layer, update_layer_ids=[a.layer - 2, a.layer - 1, a.layer],
                          param_substring="feed_forward.w2", steering_coeff=a.coeff, alpha=a.alpha, lr=a.lr,
                          max_num_batches=a.steps)
        torch.manual_seed(a.seed)
        if not masks:
            U.rmu_unlearn(model, retain_batches=move(retain), forget_batches=move(forget), config=cfg)
            return model, None
        frozen = copy.deepcopy(model)
        mk = masked.Masked(model, masks)
        with masked.patch_rmu_selection(mk.freeze()):
            U.rmu_unlearn(model, retain_batches=move(retain), forget_batches=move(forget), frozen_model=frozen, config=cfg)
        return model, mk.finalize()
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="reference target checkpoint (local dir)")
    ap.add_argument("--source", default=None, help="where --model came from (s3 uri), recorded in the metadata")
    ap.add_argument("--target", choices=("book", "calculus", "arithmetic", "algebra"), required=True,
                    help="arithmetic/algebra: depth-ladder controls (PLAN 4.7)")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--selfgen", default="work/selfgen.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=200, help="max optimizer steps (ga/graddiff/gakl) or batches (rmu)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.1, help="npo")
    ap.add_argument("--rho", type=float, default=0.01, help="npo_sam perturbation radius (paper grid 1e-3..1e-1)")
    ap.add_argument("--layer", type=int, default=7, help="rmu")
    ap.add_argument("--coeff", type=float, default=20.0, help="rmu steering coefficient")
    ap.add_argument("--alpha", type=float, default=100.0, help="rmu retain weight")
    ap.add_argument("--n-forget", type=int, default=0, help="forget items (0 = all)")
    ap.add_argument("--n-retain", type=int, default=0, help="retain items (0 = all)")
    ap.add_argument("--retain-scope", choices=("all", "general"), default="all",
                    help="general = retain only on general chat answers (depth-ladder variant without the topic anchor)")
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--mask", default=None, help="localizer scores.pt: restrict npo_full/rmu/ablate to the top units")
    ap.add_argument("--mask-frac", type=float, default=0.005, help="mask budget, fraction of all parameters")
    ap.add_argument("--mask-kinds", default="mlp,head,conv")
    ap.add_argument("--mask-lambda", type=float, default=1.0, help="forget-specific score = s_forget - lambda * s_retain")
    ap.add_argument("--mask-max-layer", type=int, default=None, help="only units in layers <= this (rmu: always --layer)")
    ap.add_argument("--mask-random", action="store_true", help="same-size random-mask control")
    a = ap.parse_args()
    if (a.mask or a.method == "ablate") and a.method not in MASKABLE:
        sys.exit(f"--mask works with {MASKABLE}")
    if a.method == "ablate" and not a.mask:
        sys.exit("ablate needs --mask")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import safetune

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    forget, retain, stats = build(a.target, Path(a.data), tok, selfgen=Path(a.selfgen) if Path(a.selfgen).is_file() else None,
                                  n_forget=a.n_forget, n_retain=a.n_retain, budget=a.batch_tokens, max_len=a.max_len,
                                  seed=a.seed, retain_scope=a.retain_scope)
    log(f"{a.name}: {a.method} on {a.target}; data {json.dumps(stats)}")
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    if a.method != "rmu" and not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    masks, mask_sel = build_mask(a, model)
    if mask_sel:
        log(f"mask: {mask_sel['n_units']} units, {mask_sel['params']} params ({'random' if a.mask_random else 'localized'}), "
            f"by kind {mask_sel['by_kind']}")
    t1 = time.time()
    model, mask_rep = run(a.method, model, a, forget, retain, masks)
    train_s = time.time() - t1
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    report = {"name": a.name, "method": a.method, "target": a.target, "config": vars(a), "data": stats,
              "train_seconds": round(train_s, 1), "finite": finite,
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device == "cuda" else None,
              "safetune": safetune.__version__, "torch": torch.__version__, "run_id": os.environ.get("RUN_ID"),
              "code_sha": os.environ.get("CODE_SHA"), "base_model": a.source or a.model,
              "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "mask": ({k: v for k, v in mask_sel.items() if k != "units"} | {"result": mask_rep}) if mask_sel else None}
    if mask_sel and a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        (Path(a.log).parent / "mask_units.json").write_text(json.dumps(mask_sel["units"]), encoding="utf-8")
    log(f"trained in {train_s / 60:.1f} min, peak {report['peak_gpu_gb']} GB, finite {finite}")
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not finite:
        sys.exit("non-finite weights; not saving")
    if not a.no_save:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        ckpt.save(model, tok, a.out, a.model, report)
        log(f"saved {a.out} ({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
