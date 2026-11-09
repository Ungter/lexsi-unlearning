"""Fit Jacobian lenses on LFM3 checkpoints with anthropics/jacobian-lens (import name ``jlens``).

The lens reads a residual ``h_l`` (output of decoder block ``l``) as ``unembed(J_l @ h_l)`` with
``J_l = E[d h_final / d h_l]`` averaged over a web-text corpus; ``h_final`` is the output of the last block
(before ``model.model.embedding_norm``). Fitted layers are 0..n_layers-2; the last block is the model output.

LFM3 layout: decoder ``model.model``, blocks ``.layers``, final norm ``.embedding_norm`` (not ``.norm``, so jlens
auto-detection fails and the layout must be passed), embedding ``.embed_tokens``, tied ``lm_head``.

Cost per prompt: one forward and ``ceil(d_model / dim_batch)`` backward passes (input grads only). ``fit_lens``
runs ``jlens.fit`` on chunks and combines them with ``JacobianLens.merge`` so a fit can stop on a time budget.

CLI:  python -m lfm3u.jspace.fit --model REF --corpus fineweb.jsonl --lens-out work/jspace/base.pt
                                 --summary results/jspace/base/fit.json [--fit-seconds 900 --min-prompts 100]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch

log = logging.getLogger("lfm3u.jspace")

#: FineWeb docs 0-499 are the held-out perplexity set; lenses are fitted on docs 500-1499 only.
FIT_DOC_RANGE = (500, 1500)
SEQ_LEN = 128


def lfm3_layout():
    """jlens Layout for Lfm3ForCausalLM (PLAN's ``Layout("model", norm="embedding_norm")`` is equivalent)."""
    from jlens import Layout

    return Layout("model", layers="layers", norm="embedding_norm", embed="embed_tokens", lm_head="lm_head")


def load_model(ref: str, device: str | None = None, dtype: torch.dtype = torch.bfloat16):
    """Load an LFM3 checkpoint (HF id or local dir) and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(ref, trust_remote_code=True, dtype=dtype).to(device).eval()
    return hf, tok


def wrap(hf_model, tokenizer, *, compile: bool = False):
    """Wrap a loaded HF LFM3 model as a jlens LensModel. Mutates the model (requires_grad off) and sets
    ``tokenizer.add_bos_token = True`` (LFM3 ships with False; raw-text prompts need the BOS sink)."""
    import jlens

    return jlens.from_hf(hf_model, tokenizer, layout=lfm3_layout(), compile=compile, force_bos=True)


def fineweb_prompts(path: str | Path, tokenizer, *, start: int = FIT_DOC_RANGE[0], end: int = FIT_DOC_RANGE[1],
                    seq_len: int = SEQ_LEN) -> list[tuple[int, str]]:
    """(doc index, text) for FineWeb docs in [start, end) that fill a ``seq_len``-token sequence (BOS included).
    jlens truncates each prompt to ``seq_len`` tokens; the text is pre-cut to bound tokenization cost."""
    if start < FIT_DOC_RANGE[0]:
        raise ValueError(f"docs below {FIT_DOC_RANGE[0]} are the held-out perplexity set")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            text = json.loads(line)["text"][: 12 * seq_len]
            if len(tokenizer(text, add_special_tokens=False).input_ids) >= seq_len - 1:
                out.append((i, text))
    return out


def auto_dim_batch(device: torch.device | str | None = None) -> int:
    """Output dims per backward pass. The forward graph of ``dim_batch`` x 128 tokens is retained (~13 GB at 64
    for LFM3-1.7B in bf16); larger batches only improve GPU utilisation, total FLOPs are fixed."""
    if not torch.cuda.is_available():
        return 8
    gb = torch.cuda.get_device_properties(device or 0).total_memory / 2**30
    return 64 if gb >= 40 else 32


def fit_lens(jm, prompts: list[str], *, chunk: int = 25, min_prompts: int = 100, max_prompts: int = 1000,
             time_budget_s: float | None = None, dim_batch: int | None = None, max_seq_len: int = SEQ_LEN,
             save_path: str | Path | None = None, save_every: int = 100):
    """Fit a lens on ``prompts`` in chunks of ``chunk`` (``jlens.fit`` + ``JacobianLens.merge``).

    Stops after ``max_prompts``, or once at least ``min_prompts`` are done and ``time_budget_s`` has elapsed.
    Saves the running lens (fp16) to ``save_path`` every ``save_every`` prompts and at the end.
    Returns ``(lens, stats)``.
    """
    import jlens
    from jlens import JacobianLens

    dim_batch = dim_batch or auto_dim_batch()
    lens, n_seen, last_save = None, 0, 0
    chunk_times: list[tuple[int, float]] = []
    t_first_100 = None
    limit = min(max_prompts, len(prompts))
    t0 = time.perf_counter()
    while n_seen < limit:
        part = prompts[n_seen: min(n_seen + chunk, limit)]
        tc = time.perf_counter()
        try:
            part_lens = jlens.fit(jm, part, dim_batch=dim_batch, max_seq_len=max_seq_len, checkpoint_path=None)
        except torch.cuda.OutOfMemoryError:
            if dim_batch <= 8:
                raise
            dim_batch //= 2
            log.warning("fit: CUDA OOM, retrying chunk with dim_batch=%d", dim_batch)
            torch.cuda.empty_cache()
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - tc
        chunk_times.append((part_lens.n_prompts, dt))
        lens = part_lens if lens is None else JacobianLens.merge([lens, part_lens])
        n_seen += len(part)
        elapsed = time.perf_counter() - t0
        if t_first_100 is None and lens.n_prompts >= 100:
            t_first_100 = elapsed
        log.info("fit: %d prompts in %.0fs (%.2fs/prompt this chunk)", lens.n_prompts, elapsed, dt / max(1, len(part)))
        if save_path and lens.n_prompts - last_save >= save_every:
            lens.save(str(save_path))
            last_save = lens.n_prompts
        if time_budget_s is not None and elapsed >= time_budget_s and lens.n_prompts >= min_prompts:
            log.info("fit: time budget %.0fs reached", time_budget_s)
            break
    if lens is None:
        raise ValueError("no prompts to fit on")
    if save_path:
        lens.save(str(save_path))
    total = time.perf_counter() - t0
    stats = {
        "n_prompts": lens.n_prompts,
        "n_prompts_seen": n_seen,
        "dim_batch": dim_batch,
        "backward_passes_per_prompt": math.ceil(jm.d_model / dim_batch),
        "fit_seconds": round(total, 1),
        "seconds_per_prompt": round(total / lens.n_prompts, 3),
        "seconds_first_100_prompts": None if t_first_100 is None else round(t_first_100, 1),
        "chunk_seconds": [[n, round(t, 1)] for n, t in chunk_times],
    }
    return lens, stats


def lens_diagnostics(lens) -> dict:
    """Per-layer ||J_l||_F / sqrt(d) and mean diag(J_l) (diag -> ~1 near the output for a residual stream)."""
    d = lens.d_model
    return {
        str(layer): {"fro_over_sqrt_d": round(J.norm().item() / math.sqrt(d), 4),
                     "diag_mean": round(J.diagonal().mean().item(), 4)}
        for layer, J in sorted(lens.jacobians.items())
    }


def env_info() -> dict:
    import transformers

    info = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        from importlib.metadata import version

        info["jlens"] = version("jlens")
    except Exception:  # noqa: BLE001
        info["jlens"] = "source checkout"
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--corpus", required=True, help="FineWeb jsonl (fields id,url,text)")
    ap.add_argument("--lens-out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--fit-seconds", type=float, default=None)
    ap.add_argument("--min-prompts", type=int, default=100)
    ap.add_argument("--max-prompts", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--dim-batch", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    hf, tok = load_model(a.model)
    jm = wrap(hf, tok)
    docs = fineweb_prompts(a.corpus, tok)
    Path(a.lens_out).parent.mkdir(parents=True, exist_ok=True)
    lens, stats = fit_lens(jm, [t for _, t in docs], chunk=a.chunk, min_prompts=a.min_prompts,
                           max_prompts=a.max_prompts, time_budget_s=a.fit_seconds, dim_batch=a.dim_batch or None,
                           save_path=a.lens_out)
    summary = {"model": a.model, "lens_file": os.path.basename(a.lens_out), **stats,
               "doc_index_range": [docs[0][0], docs[stats["n_prompts_seen"] - 1][0]], "env": env_info(),
               "diagnostics": lens_diagnostics(lens)}
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
