"""Masked unlearning (PLAN 5.3, 5.4): only the input columns of the selected units may change.

`Masked(model, masks)` freezes every parameter, re-enables gradients on the tensors that contain selected columns and
multiplies their gradients by a 0/1 column mask. `finalize()` then restores every unmasked element to its original
value (AdamW's decoupled weight decay touches whole tensors) and reports how many elements actually changed.
`patch_rmu_selection` makes SafeTune's rmu_unlearn train exactly these tensors instead of w2 of its update layers.
`ablate` zeroes the selected columns (mean-free zero ablation, no training).
"""
from __future__ import annotations

import contextlib


def _dense(p, cols):
    import torch

    m = torch.zeros_like(p, dtype=p.dtype)
    m[:, cols.to(p.device)] = 1
    return m


class Masked:
    def __init__(self, model, masks: dict):
        params = dict(model.named_parameters())
        missing = sorted(set(masks) - set(params))
        if missing:
            raise KeyError(f"mask names not in the model: {missing[:3]}")
        self.model = model
        self.params = {n: params[n] for n in masks}
        self.masks = {n: _dense(self.params[n], c) for n, c in masks.items()}
        self.allowed = sum(self.params[n].shape[0] * len(c) for n, c in masks.items())
        self.orig = {n: p.detach().clone() for n, p in self.params.items()}
        self.handles = []

    def freeze(self) -> list:
        for p in self.model.parameters():
            p.requires_grad_(False)
        for n, p in self.params.items():
            p.requires_grad_(True)
            self.handles.append(p.register_hook(lambda g, m=self.masks[n]: g * m.to(g.dtype)))
        return list(self.params.values())

    def finalize(self) -> dict:
        import torch

        for h in self.handles:
            h.remove()
        self.handles = []
        changed = outside = 0
        max_delta = 0.0
        with torch.no_grad():
            for n, p in self.params.items():
                o, m = self.orig[n].to(p.dtype), self.masks[n].to(p.dtype)
                delta = p - o
                outside += int(((delta * (1 - m)) != 0).sum())
                p.copy_(o + delta * m)
                d = (p - o).abs()
                changed += int((d > 0).sum())
                max_delta = max(max_delta, float(d.max()))
                p.requires_grad_(False)
        total = sum(p.numel() for p in self.model.parameters())
        self.orig = {}
        return {"tensors": len(self.params), "mask_elements": self.allowed, "changed_elements": changed,
                "changed_frac_of_model": round(changed / total, 6), "reset_outside_mask": outside,
                "max_abs_delta": max_delta}


@contextlib.contextmanager
def patch_rmu_selection(params: list):
    """Inside the block, safetune's rmu_unlearn trains `params` (requires_grad already set by Masked.freeze)."""
    import safetune.interventions.unlearn.rmu as rmu

    orig = rmu._select_params
    rmu._select_params = lambda model, cfg: params
    try:
        yield
    finally:
        rmu._select_params = orig


def ablate(model, masks: dict) -> dict:
    import torch

    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for name, cols in masks.items():
            p = params[name]
            p[:, cols.to(p.device)] = 0
            n += p.shape[0] * len(cols)
    return {"tensors": len(masks), "zeroed_elements": n}
