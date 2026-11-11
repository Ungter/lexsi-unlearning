"""Edit math for weight surgery. Every edit is a delta on one tensor, kept in fp32:
  lowrank  {"U": [rows, k], "V": [cols, k]}            delta = U @ V.T
  cols     {"cols": LongTensor, "delta": [rows, n]}    delta on whole input columns (units)
  flat     {"idx": LongTensor, "delta": [n]}           delta on flattened elements

Direction removal under sandwich norms. LFM3 adds post_norm(W a) to the stream, with post_norm(y) = g * y / rms(y)
(Lfm3RMSNorm). The stream component along r of that write is r . (g * y) / rms(y) = (g * r) . y / rms(y), so the write
is free of r exactly when y is orthogonal to u = (g * r) / |g * r|, not to r. `ablate_out` projects the writer's rows
off u: W' = (I - u u^T) W. Readers see pre_norm(x) = g_in * x / rms(x); `ablate_in` makes a reader blind to the stream
component along r: W' = W - [W (g_in * r)] (r / g_in)^T, since (r / g_in) . (g_in * x) = r . x. (rms(x) still sees it.)
"""
from __future__ import annotations

import re

WRITERS = {"feed_forward.w2": "post_ffn_norm", "self_attn.o_proj": "post_operator_norm", "conv.out_proj": "post_operator_norm"}
READERS = {"feed_forward.w1": "ffn_norm", "feed_forward.w3": "ffn_norm", "self_attn.q_proj": "operator_norm",
           "self_attn.k_proj": "operator_norm", "self_attn.v_proj": "operator_norm", "self_attn.gate_proj": "operator_norm",
           "conv.in_proj": "operator_norm"}
_NAME = re.compile(r"^(model\.layers\.\d+)\.(.+)\.weight$")


def _norm_for(param: str, table: dict) -> str | None:
    m = _NAME.match(param)
    if not m or m.group(2) not in table:
        return None
    return f"{m.group(1)}.{table[m.group(2)]}.weight"


def writer_norm(param: str) -> str | None:
    """Name of the post-norm gain that follows this residual-stream writer (None if param is not a writer)."""
    return _norm_for(param, WRITERS)


def reader_norm(param: str) -> str | None:
    """Name of the pre-norm gain in front of this residual-stream reader (None if param is not a reader)."""
    return _norm_for(param, READERS)


def unit(v):
    return v / v.norm()


def sandwich_direction(r, g):
    """u = (g * r) / |g * r|: the writer-output direction whose removal zeroes the post-norm write along r."""
    return unit(g * r)


def ablate_out(W, r, g=None):
    """Low-rank delta (U, V) with W + U V^T = (I - u u^T) W; u = unit(g * r), or unit(r) without a norm gain."""
    u = sandwich_direction(r, g) if g is not None else unit(r)
    return u[:, None].clone(), -(W.T @ u)[:, None]


def ablate_out_stream(W, r, g, mass: float | None = None, kappa: float = 3.0):
    """Low-rank delta (U, V) that removes r from the writer's post-norm write by an ORTHOGONAL projection in stream
    space, as the activation hook does: g * (W' a) = (I - r_m r^T / |r_m|^2)(g * W a), i.e.
    W' = W - (r_m / g)(g * r)^T W / |r_m|^2. The write along r is exactly zero for any mask m.

    r_m is r restricted to the highest-gain output dims. Dims with a near-zero gain write almost nothing, and dividing
    by their gain blows up the pre-norm output (the post-norm then shrinks everything else). The mask is the largest
    set of top-gain dims whose pre-norm amplification kappa = |r_m / g| |g * r| / |r_m|^2 stays <= `kappa` (kappa = 1
    is the size of the pre-norm change made by `ablate_out`), or, if `mass` is given, the top-gain dims holding that
    share of |r|^2.

    Compare `ablate_out` (projection orthogonal in PRE-norm space): it also zeroes the write along r, but changes the
    stream along g^2 * r instead of r, which leaves the feature readable downstream (docs/SURGERY.md §6)."""
    import torch

    r = unit(r)
    order = g.abs().argsort(descending=True)
    r2 = r[order].pow(2).cumsum(0)
    if mass is not None:
        k = int((r2 < mass * r2[-1]).sum()) + 1
    else:
        kap = (r[order] / g[order]).pow(2).cumsum(0).sqrt() * (g * r).norm() / r2.clamp(min=1e-12)
        ok = ((kap <= kappa) & (r2 >= 0.5 * r2[-1])).nonzero()
        k = int(ok.max()) + 1 if len(ok) else int((r2 < 0.5 * r2[-1]).sum()) + 1
    m = torch.zeros_like(r)
    m[order[:k]] = 1
    rm = r * m
    U = torch.where(m.bool(), -rm / g, torch.zeros_like(r)) / rm.pow(2).sum()
    return U[:, None], (W.T @ (g * r))[:, None]


def ablate_in(W, r, g=None):
    """Low-rank delta (U, V) that makes the reader W blind to the stream component along r (see module docstring)."""
    r = unit(r)
    if g is None:
        return -(W @ r)[:, None], r[:, None].clone()
    return -(W @ (g * r))[:, None], (r / g)[:, None]


def scale_cols(W, cols, factor: float):
    """Delta that multiplies the input columns `cols` by `factor`."""
    return (factor - 1.0) * W[:, cols]


def dense(spec: dict, shape, device=None):
    """The fp32 delta tensor of an edit spec."""
    import torch

    d = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    if spec["kind"] == "lowrank":
        d += spec["U"].to(device, torch.float32) @ spec["V"].to(device, torch.float32).T
    elif spec["kind"] == "cols":
        d[:, spec["cols"].to(device)] += spec["delta"].to(device, torch.float32)
    elif spec["kind"] == "flat":
        d.view(-1)[spec["idx"].to(device)] += spec["delta"].to(device, torch.float32)
    else:
        raise ValueError(spec["kind"])
    return d
