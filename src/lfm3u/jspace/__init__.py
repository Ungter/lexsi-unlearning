"""J-space tools (PLAN Phase 3) on top of anthropics/jacobian-lens (``pip install --no-deps`` the repo; import ``jlens``).

fit.py       load LFM3, wrap it for jlens (final norm = model.model.embedding_norm), fit lenses on FineWeb 500-1499
readout.py   rank_of(lens, model, tokenizer, prompt, target_token_ids, positions) -> per-layer ranks; heatmaps
inject.py    concept injection v = J_l^T u_t (add / swap hooks), alpha sweeps, country->capital validation
validate.py  PLAN 3.2 validation prompt sets and the knowmem "gone or just hidden" readout
Driver: scripts/04_jspace.py; GPU job: scripts/vast/jobs/jspace.sh.
"""
from lfm3u.jspace.fit import fineweb_prompts, fit_lens, lfm3_layout, load_model, wrap
from lfm3u.jspace.inject import concept_direction
from lfm3u.jspace.readout import first_token_variants, layer_logits, plot_rank_heatmap, rank_of

# The injection context manager is lfm3u.jspace.inject.inject (not re-exported: it would shadow the submodule).
__all__ = ["concept_direction", "fineweb_prompts", "first_token_variants", "fit_lens", "layer_logits",
           "lfm3_layout", "load_model", "plot_rank_heatmap", "rank_of", "wrap"]
