"""Jacobian-lens readouts: per-layer rank of target tokens, lens/output agreement, layer x prompt heatmaps.

Rank convention: 1-based over the full vocabulary (rank 1 = top-1; "top-10" means rank <= 10). For several target
ids (e.g. "Paris" and " Paris") the rank is the minimum over them. Rank arrays are indexed by layer: rows
0..n_layers-2 are lens readouts of block outputs, row n_layers-1 is the model's actual next-token distribution.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def as_lens_model(model, tokenizer=None):
    """Accept a jlens LensModel or a HF LFM3 model (wrapped with the LFM3 layout)."""
    if hasattr(model, "unembed") and hasattr(model, "layers") and hasattr(model, "n_layers"):
        return model
    from lfm3u.jspace.fit import wrap

    if tokenizer is None:
        raise ValueError("a tokenizer is needed to wrap a HF model")
    return wrap(model, tokenizer)


def lens_to(lens, device) -> None:
    """Move the J_l matrices to ``device`` once (``JacobianLens.transport`` otherwise copies them every call)."""
    lens.jacobians = {layer: J.to(device) for layer, J in lens.jacobians.items()}


def encode(tokenizer, prompt) -> torch.Tensor:
    """``[1, T]`` input ids. Strings that already start with the BOS token (chat-template output) are tokenized
    without special tokens; other strings get the BOS prepended."""
    if isinstance(prompt, torch.Tensor):
        return prompt.reshape(1, -1)
    if not isinstance(prompt, str):
        return torch.tensor([list(prompt)])
    bos = getattr(tokenizer, "bos_token", None)
    if bos and prompt.startswith(bos):
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if getattr(tokenizer, "bos_token_id", None) is not None:
            ids = [tokenizer.bos_token_id, *ids]
    return torch.tensor([ids])


@torch.no_grad()
def layer_logits(jm, lens, input_ids: torch.Tensor, *, positions: Sequence[int] | None = (-1,),
                 layers: Sequence[int] | None = None, use_jacobian: bool = True) -> dict[int, torch.Tensor]:
    """``{layer: [n_positions, vocab] fp32}`` for the lens layers plus the final layer (model output).
    ``use_jacobian=False`` gives the plain logit lens for the same layers."""
    from jlens import ActivationRecorder

    final = jm.n_layers - 1
    layers = list(lens.source_layers if layers is None else layers)
    record = sorted(set(layers) | {final})
    with ActivationRecorder(jm.layers, at=record) as rec:
        jm.forward(input_ids.to(jm.input_device))
        acts = {i: rec.activations[i].detach() for i in record}

    def sel(layer: int) -> torch.Tensor:
        h = acts[layer][0]
        return (h if positions is None else h[list(positions)]).float()

    out = {}
    for layer in layers:
        if layer == final:
            continue
        h = sel(layer)
        if use_jacobian:
            h = lens.transport(h, layer)
        out[layer] = jm.unembed(h).float()
    out[final] = jm.unembed(sel(final)).float()
    return out


def ranks_from_logits(logits: torch.Tensor, target_ids: Sequence[int]) -> torch.Tensor:
    """1-based rank (min over ``target_ids``) for each row of ``[P, V]`` logits -> ``[P]``."""
    tgt = logits[:, list(target_ids)]  # [P, K]
    ranks = (logits[:, None, :] > tgt[:, :, None]).sum(-1) + 1  # [P, K]
    return ranks.min(dim=1).values


def rank_of(lens, model, tokenizer, prompt, target_token_ids: Sequence[int], positions: Sequence[int] = (-1,), *,
            use_jacobian: bool = True) -> np.ndarray:
    """Per-layer rank of the target token(s) at ``positions``.

    Args:
        lens: fitted ``jlens.JacobianLens``.
        model: jlens LensModel, or a HF LFM3 model (wrapped on the fly).
        tokenizer: the model's tokenizer.
        prompt: text (chat-template text starting with BOS is used as is), or token ids.
        target_token_ids: one or more vocabulary ids; the rank is the minimum over them.
        positions: token positions (negative = from the end).
        use_jacobian: False = plain logit lens baseline.

    Returns:
        int array ``[n_layers, len(positions)]``; row ``l`` is the lens rank at block ``l`` (-1 if the lens has no
        J for that layer); the last row is the rank in the model's own output distribution.
    """
    jm = as_lens_model(model, tokenizer)
    ids = encode(tokenizer, prompt)
    logits = layer_logits(jm, lens, ids, positions=positions, use_jacobian=use_jacobian)
    out = np.full((jm.n_layers, len(positions)), -1, dtype=np.int64)
    for layer, lg in logits.items():
        out[layer] = ranks_from_logits(lg, target_token_ids).cpu().numpy()
    return out


def first_token_variants(tokenizer, word: str, *, capitalize: bool = True, min_chars: int = 3) -> list[int]:
    """First-token ids of ``word`` and ``" " + word`` (plus capitalised forms).

    Drops first tokens that decode to whitespace only (LFM3 splits " 6" into " " + "6") and fragments shorter than
    ``min_chars`` characters unless they are the whole word (no-space "Hermione" starts with "H", which would make
    any H-word a hit). If every form is dropped, falls back to the longest first token (callers can flag it)."""
    forms = [word, " " + word]
    if capitalize and word[:1].islower():
        forms += [word.capitalize(), " " + word.capitalize()]
    ids: list[int] = []
    fallback: list[tuple[int, int]] = []
    for f in forms:
        toks = tokenizer(f, add_special_tokens=False).input_ids
        if not toks:
            continue
        piece = tokenizer.decode([toks[0]]).strip()
        if not piece:
            continue
        fallback.append((len(piece), toks[0]))
        if (len(piece) >= min_chars or piece.lower() == word.strip().lower()) and toks[0] not in ids:
            ids.append(toks[0])
    if not ids and fallback:
        ids = [max(fallback)[1]]
    return ids


@torch.no_grad()
def agreement_with_output(jm, lens, tokenizer, texts: Sequence[str], *, max_seq_len: int = 128,
                          skip_first: int = 16) -> dict:
    """Per-layer top-1 agreement between the lens readout and the model's own next-token prediction, over all
    positions >= ``skip_first`` of ``texts``; J-lens vs the plain logit lens."""
    final = jm.n_layers - 1
    hits = {"jlens": np.zeros(final), "logit_lens": np.zeros(final)}
    n = 0
    for text in texts:
        ids = encode(tokenizer, text)[:, :max_seq_len]
        pos = list(range(skip_first, ids.shape[1]))
        if not pos:
            continue
        for key, use_j in (("jlens", True), ("logit_lens", False)):
            lg = layer_logits(jm, lens, ids, positions=pos, use_jacobian=use_j)
            top_out = lg[final].argmax(-1)
            for layer in lens.source_layers:
                hits[key][layer] += (lg[layer].argmax(-1) == top_out).sum().item()
        n += len(pos)
    return {k: [round(v / max(n, 1), 4) for v in arr] for k, arr in hits.items()} | {"n_positions": n}


def plot_rank_heatmap(ranks: np.ndarray, path: str | Path, *, prompt_labels: Sequence[str] | None = None,
                      title: str = "", max_rank: int | None = None) -> None:
    """Save a layer x prompt heatmap of log10(rank) (``ranks``: ``[n_layers, n_prompts]``, 1-based)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.asarray(ranks, dtype=float)
    r[r < 1] = np.nan
    n_layers, n_prompts = r.shape
    vmax = np.log10(max_rank) if max_rank else np.nanmax(np.log10(r)) if np.isfinite(r).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * n_prompts + 2), max(4, 0.25 * n_layers + 1.5)))
    im = ax.imshow(np.log10(r), aspect="auto", origin="lower", cmap="viridis_r", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.axhline(n_layers - 1.5, color="white", lw=1)
    ax.set_ylabel(f"layer (row {n_layers - 1} = model output)")
    ax.set_xlabel("prompt")
    if prompt_labels is not None and n_prompts <= 120:
        ax.set_xticks(range(n_prompts))
        ax.set_xticklabels([s[:18] for s in prompt_labels], rotation=90, fontsize=6)
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10 rank (0 = top-1, 1 = rank 10)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
