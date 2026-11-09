"""J-space concept injection.

Concept direction for token t at layer l: the pullback of its unembedding direction through the lens,
``v = J_l^T u_t`` with ``u_t = W_U[t] * g`` (``g`` = gain of the final RMSNorm, ``model.model.embedding_norm``,
which sits between the last block and the tied LM head). ``v`` is unit-normalised and added to the output of block
``l`` as ``alpha * ||h_p|| * v_hat`` at the chosen positions (scale relative to each position's residual norm,
as in the paper's steering recipe: "unit-normalized transpose row ... scaled by the residual norm times a strength").

jlens ships no injection/steering code (only fit/apply/vis), so this module implements it with forward hooks.

Modes:
  add    h_p += alpha * ||h_p|| * v_hat(to)                         (PLAN 3.4; default at the last prompt position)
  swap   exchange the coordinates of h_p along v_hat(from) and v_hat(to), the swapped-in one scaled by alpha
         (paper-style lens "swap"; intended for a band of layers and every prompt position)
"""
from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch


def unembed_direction(jm, token_id: int) -> torch.Tensor:
    """Readout direction of ``token_id`` in the final-residual basis: ``W_U[t] * g`` (fp32)."""
    w = jm._lm_head.weight[token_id].float()
    g = getattr(jm._final_norm, "weight", None)
    return w * g.float() if g is not None else w


def concept_direction(lens, jm, token_ids: int | Sequence[int], layer: int, *, normalize: bool = True) -> torch.Tensor:
    """``v = J_l^T u_t`` (mean over ``token_ids`` of the unit pullbacks when several are given)."""
    ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
    J = lens.jacobians[layer].float()
    vs = []
    for t in ids:
        v = J.T @ unembed_direction(jm, t).to(J.device)
        vs.append(v / v.norm())
    v = torch.stack(vs).mean(0)
    return v / v.norm() if normalize else v


def _resolve(positions: Sequence[int] | None, seq_len: int) -> list[int] | slice:
    if positions is None:
        return slice(1, None)  # every position after BOS
    return [p % seq_len for p in positions]


@contextlib.contextmanager
def inject(jm, edits: dict[int, dict], *, positions: Sequence[int] | None = (-1,), seq_len: int | None = None):
    """Context manager adding forward hooks on ``jm.layers``.

    Args:
        edits: ``{layer: {"mode": "add", "to": v_hat, "alpha": a}}`` or
               ``{layer: {"mode": "swap", "from": a_hat, "to": b_hat, "alpha": a}}``.
        positions: positions to edit (negative = from the end of the sequence the hook sees; pass ``seq_len`` to
            pin them to a prompt length when generating without a cache); ``None`` = every position after BOS.
    """
    handles = []

    def make_hook(spec: dict):
        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            T = hidden.shape[1]
            idx = _resolve(positions, seq_len or T)
            if isinstance(idx, list):
                idx = [p for p in idx if p < T]
                if not idx:
                    return output
            h = hidden[:, idx, :].float()
            b = spec["to"].to(h.device, torch.float32)
            alpha = float(spec.get("alpha", 1.0))
            if spec.get("mode", "add") == "add":
                h_new = h + alpha * h.norm(dim=-1, keepdim=True) * b
            else:
                a = spec["from"].to(h.device, torch.float32)
                ca, cb = h @ a, h @ b
                h_new = h + (alpha * ca - cb)[..., None] * b + (cb - ca)[..., None] * a
            new = hidden.clone()
            new[:, idx, :] = h_new.to(hidden.dtype)
            return new if torch.is_tensor(output) else (new, *output[1:])

        return hook

    try:
        for layer, spec in edits.items():
            handles.append(jm.layers[layer].register_forward_hook(make_hook(spec)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def next_token_logits(jm, input_ids: torch.Tensor) -> torch.Tensor:
    """Model next-token logits at the last position, ``[V]`` fp32 (hooks on ``jm.layers`` apply). The decoder's
    ``last_hidden_state`` is already normed by ``embedding_norm``, so only the LM head is applied."""
    out = jm.forward(input_ids.to(jm.input_device))
    return jm._lm_head(out.last_hidden_state[:, -1]).float()[0]


@torch.no_grad()
def greedy(jm, input_ids: torch.Tensor, n_tokens: int = 6) -> list[int]:
    """Greedy continuation by full recompute (no KV cache, so position-pinned hooks stay valid)."""
    ids = input_ids.to(jm.input_device)
    new = []
    for _ in range(n_tokens):
        t = int(next_token_logits(jm, ids).argmax())
        new.append(t)
        ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
    return new


def _rank(logits: torch.Tensor, t: int) -> int:
    return int((logits > logits[t]).sum().item()) + 1


@torch.no_grad()
def alpha_sweep(jm, lens, tokenizer, prompt_ids: torch.Tensor, *, to_token: int, watch: dict[str, int],
                layers: Sequence[int] | Sequence[Sequence[int]], alphas: Sequence[float], mode: str = "add",
                from_token: int | None = None, positions: Sequence[int] | None = (-1,)) -> list[dict]:
    """Run the prompt under injection for each (layer or layer band, alpha). Records the top-1 next token and
    the probability / rank of every ``watch`` token. ``layers`` items may be ints or bands (lists of ints)."""
    rows = []
    T = prompt_ids.shape[1]
    for spec_layers in layers:
        band = [spec_layers] if isinstance(spec_layers, int) else list(spec_layers)
        dirs = {l: concept_direction(lens, jm, to_token, l) for l in band}
        froms = {l: concept_direction(lens, jm, from_token, l) for l in band} if mode == "swap" else {}
        for alpha in alphas:
            edits = {l: {"mode": mode, "to": dirs[l], "from": froms.get(l), "alpha": alpha} for l in band}
            with inject(jm, edits if alpha or mode == "swap" else {}, positions=positions, seq_len=T):
                lg = next_token_logits(jm, prompt_ids)
            p = torch.softmax(lg, -1)
            top = int(lg.argmax())
            rows.append({"mode": mode, "layers": band, "alpha": alpha, "top1": tokenizer.decode([top]),
                         **{f"p_{k}": round(p[t].item(), 4) for k, t in watch.items()},
                         **{f"rank_{k}": _rank(lg, t) for k, t in watch.items()}})
    return rows


def capital_swap(jm, lens, tokenizer, pairs: Sequence[dict], *, template: str = "The capital of {} is the city of",
                 layers_single: Sequence[int], bands: Sequence[Sequence[int]], alphas_add: Sequence[float],
                 alphas_swap: Sequence[float]) -> dict:
    """Country -> capital validation. For each pair {src, dst, src_cap, dst_cap} the prompt names ``src``; we
    inject ``dst`` and check whether the next token becomes ``dst_cap`` (not merely ``dst``).
    Modes: add@last (PLAN), add@all positions, swap over layer bands at all positions (paper-style)."""
    from lfm3u.jspace.readout import encode

    def tid(w: str) -> int:
        return tokenizer(" " + w, add_special_tokens=False).input_ids[0]

    results = []
    for pr in pairs:
        ids = encode(tokenizer, template.format(pr["src"]))
        watch = {"dst_cap": tid(pr["dst_cap"]), "src_cap": tid(pr["src_cap"]), "dst": tid(pr["dst"])}
        clean = next_token_logits(jm, ids)
        runs = {
            "add_last": alpha_sweep(jm, lens, tokenizer, ids, to_token=watch["dst"], watch=watch,
                                    layers=layers_single, alphas=alphas_add, mode="add", positions=(-1,)),
            "add_all": alpha_sweep(jm, lens, tokenizer, ids, to_token=watch["dst"], watch=watch,
                                   layers=layers_single, alphas=alphas_add, mode="add", positions=None),
            "swap_band": alpha_sweep(jm, lens, tokenizer, ids, to_token=watch["dst"], from_token=tid(pr["src"]),
                                     watch=watch, layers=bands, alphas=alphas_swap, mode="swap", positions=None),
        }
        summary = {}
        for name, rows in runs.items():
            if not rows:
                summary[name] = {"success": False, "n_settings": 0}
                continue
            ok = [r for r in rows if r["rank_dst_cap"] == 1]
            best = max(rows, key=lambda r: r["p_dst_cap"])
            summary[name] = {"success": bool(ok), "n_settings": len(rows), "n_success": len(ok),
                             "best": best,
                             "success_settings": [{"layers": r["layers"], "alpha": r["alpha"]} for r in ok][:20]}
        # qualitative continuation at the best add_last setting
        b = summary["add_last"]["best"]
        v = concept_direction(lens, jm, watch["dst"], b["layers"][0])
        with inject(jm, {b["layers"][0]: {"mode": "add", "to": v, "alpha": b["alpha"]}}, positions=(-1,),
                    seq_len=ids.shape[1]):
            cont = tokenizer.decode(greedy(jm, ids, 6))
        results.append({**pr, "prompt": template.format(pr["src"]),
                        "clean_top1": tokenizer.decode([int(clean.argmax())]),
                        "clean_rank_src_cap": _rank(clean, watch["src_cap"]),
                        "clean_rank_dst_cap": _rank(clean, watch["dst_cap"]),
                        "summary": summary, "best_add_last_continuation": cont, "runs": runs})
    agg = {m: sum(r["summary"][m]["success"] for r in results) for m in ("add_last", "add_all", "swap_band")}
    return {"template": template, "n_pairs": len(results), "pairs_with_success": agg, "pairs": results}
