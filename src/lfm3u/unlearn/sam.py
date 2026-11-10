"""SAM-enhanced NPO-KL (Fan et al., ICML 2025, arXiv 2502.05374, Algorithm A1).

Robust unlearning as min_theta max_{||delta||_2 <= rho} l_f(theta + delta) + lambda * l_r(theta): the adversary moves the
weights in the direction that most increases the forget loss (towards relearning), and the update minimises the forget
loss there. First-order inner step (their eq. 4): delta = rho * grad l_f / ||grad l_f||_2 over all trainable parameters.

Per step: (1) grad of the NPO forget loss at theta; (2) theta += delta; (3) forget loss at theta + delta, backward;
(4) theta -= delta; (5) KL(model || reference) on the retain batch at theta, backward (accumulates); (6) optimizer step.
The NPO and KL terms are SafeTune's own (npo_forget_loss, get_batch_loss, _retain_kl), so rho = 0 is SafeTune's
npo_unlearn(variant="npo_KL") with one extra forward/backward. The reference logits on the forget batch are computed once
per step and reused for both forget passes.
"""
from __future__ import annotations

import time


def log(msg: str) -> None:
    print(f"[sam {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sam_npo_unlearn(model, forget_batches, retain_batches, reference, *, rho: float, beta: float = 0.1, lr: float = 1e-5,
                    epochs: int = 1, max_steps: int | None = None, weight_decay: float = 0.01, npo_coeff: float = 1.0,
                    kl_coeff: float = 1.0):
    import torch
    from safetune.interventions.unlearn.npo import _retain_kl, get_batch_loss, npo_forget_loss

    def logits(m, b):
        return m(**{k: v for k, v in b.items() if k != "labels"}).logits

    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = 0
    for epoch in range(max(1, epochs)):
        for fb, rb in zip(forget_batches, retain_batches):
            if max_steps and steps >= max_steps:
                return model
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                ref_f = get_batch_loss(logits(reference, fb), fb["labels"])
            fl = npo_forget_loss(get_batch_loss(logits(model, fb), fb["labels"]), ref_f, beta)
            if rho > 0:
                grads = torch.autograd.grad(fl, params, allow_unused=True)
                norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None))
                scale = rho / (norm + 1e-12)
                deltas = [None if g is None else g * scale.to(g.dtype) for g in grads]
                del grads
                with torch.no_grad():
                    for p, d in zip(params, deltas):
                        if d is not None:
                            p.add_(d)
                fl_adv = npo_forget_loss(get_batch_loss(logits(model, fb), fb["labels"]), ref_f, beta)
                (npo_coeff * fl_adv).backward()
                with torch.no_grad():
                    for p, d in zip(params, deltas):
                        if d is not None:
                            p.sub_(d)
                del deltas
            else:
                fl_adv = fl
                (npo_coeff * fl).backward()
            with torch.no_grad():
                ref_r = logits(reference, rb)
            rl = _retain_kl(logits(model, rb), ref_r)
            (kl_coeff * rl).backward()
            opt.step()
            steps += 1
            if steps % 25 == 0:
                log(f"step {steps}: forget {float(fl):.4f} (adv {float(fl_adv):.4f}) retain_kl {float(rl):.4f}")
        log(f"epoch {epoch + 1} done ({steps} steps)")
    return model
