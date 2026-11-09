"""Phase 3 driver: fit a Jacobian lens on one model and run the J-space readouts (PLAN 3.1-3.4).

  python scripts/05_jspace.py --name base --model LiquidAI/LFM3-1.7B --corpus work/fineweb.jsonl \
      --questions work/knowmem.jsonl --stages fit,agree,validate,hidden,inject [--fit-seconds 900]

Stages (each after "fit" is independent; a failing stage is logged and the others still run):
  fit       fit the lens on FineWeb docs 500-1499 (128-token sequences) -> LENS_DIR/NAME/lens.pt + fit.json
            (or --lens PATH to load an existing lens instead)
  agree     per-layer top-1 agreement of lens vs model output on 20 FineWeb docs not used for fitting
  validate  PLAN 3.2 prompt sets -> validation.json + heatmaps
  hidden    knowmem forget readout (gold first-token rank per layer) -> hidden.json + heatmap
  inject    country -> capital injection sweep -> inject.json
Outputs: OUT/NAME/*.json|png (small), LENS_DIR/NAME/lens.pt (large; uploaded separately).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from lfm3u.jspace import fit as jfit  # noqa: E402
from lfm3u.jspace import inject as jinj  # noqa: E402
from lfm3u.jspace import readout as jread  # noqa: E402
from lfm3u.jspace import validate as jval  # noqa: E402

log = logging.getLogger("jspace")

CAPITAL_PAIRS = [
    {"src": "France", "dst": "China", "src_cap": "Paris", "dst_cap": "Beijing"},
    {"src": "Germany", "dst": "Japan", "src_cap": "Berlin", "dst_cap": "Tokyo"},
    {"src": "Italy", "dst": "Russia", "src_cap": "Rome", "dst_cap": "Moscow"},
    {"src": "Spain", "dst": "Egypt", "src_cap": "Madrid", "dst_cap": "Cairo"},
    {"src": "Japan", "dst": "France", "src_cap": "Tokyo", "dst_cap": "Paris"},
]


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default=None, help="where --model came from (e.g. its s3:// URI), recorded in fit.json")
    ap.add_argument("--corpus", default="work/fineweb.jsonl")
    ap.add_argument("--questions", default="work/knowmem.jsonl")
    ap.add_argument("--out", default="results/jspace")
    ap.add_argument("--lens-dir", default="work/jspace")
    ap.add_argument("--lens", default=None, help="load this lens instead of fitting")
    ap.add_argument("--stages", default="fit,agree,validate,hidden,inject")
    ap.add_argument("--fit-seconds", type=float, default=None)
    ap.add_argument("--min-prompts", type=int, default=100)
    ap.add_argument("--max-prompts", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--dim-batch", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stages = a.stages.split(",")
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    lens_dir = Path(a.lens_dir) / a.name
    lens_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    hf, tok = jfit.load_model(a.model, device=a.device)
    jm = jfit.wrap(hf, tok)
    log.info("loaded %s in %.0fs: %r", a.model, time.time() - t, jm)
    docs = jfit.fineweb_prompts(a.corpus, tok)
    log.info("%d FineWeb docs in [500, 1500) fill %d tokens", len(docs), jfit.SEQ_LEN)

    if a.lens:
        from jlens import JacobianLens

        lens = JacobianLens.load(a.lens)
        n_fit = lens.n_prompts
    elif "fit" in stages:
        lens, stats = jfit.fit_lens(jm, [d for _, d in docs], chunk=a.chunk, min_prompts=a.min_prompts,
                                    max_prompts=a.max_prompts, time_budget_s=a.fit_seconds,
                                    dim_batch=a.dim_batch or None, save_path=lens_dir / "lens.pt")
        n_fit = stats["n_prompts_seen"]
        summary = {"name": a.name, "model": a.model, "source": a.source or a.model, "lens_file": "lens.pt", **stats,
                   "doc_index_range": [docs[0][0], docs[n_fit - 1][0]], "env": jfit.env_info(),
                   "lens_file_mb": round((lens_dir / "lens.pt").stat().st_size / 2**20, 1),
                   "diagnostics": jfit.lens_diagnostics(lens)}
        dump(summary, out / "fit.json")
        dump(summary, lens_dir / "fit.json")
    else:
        sys.exit("no lens: pass --lens or include the fit stage")
    jread.lens_to(lens, jm.input_device)

    def stage(name, fn):
        if name not in stages:
            return
        t0 = time.time()
        try:
            fn()
            log.info("stage %s done in %.0fs", name, time.time() - t0)
        except Exception:  # noqa: BLE001
            log.error("stage %s failed:\n%s", name, traceback.format_exc())
            (out / f"{name}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    def agree():
        held = [d for _, d in docs[n_fit:]][-20:]
        res = jread.agreement_with_output(jm, lens, tok, held)
        res["docs"] = [i for i, _ in docs[n_fit:]][-20:]
        dump(res, out / "agreement.json")

    def validate():
        res = jval.run_validation(jm, lens, tok)
        dump(res, out / "validation.json")
        for set_name, s in res["sets"].items():
            labels = [p["name"] for p in res["prompts"] if p["set"] == set_name]
            try:
                jread.plot_rank_heatmap(s["ranks"], out / f"validation_{set_name}.png", prompt_labels=labels,
                                        title=f"{a.name}: rank of intermediate ({set_name}), last prompt token",
                                        max_rank=len(tok))
            except ImportError as e:
                log.warning("heatmap skipped: %s", e)
        log.info("validation hit@10: %s overall=%.2f",
                 {k: v["hit_rate"] for k, v in res["sets"].items()}, res["overall_hit_rate"])

    def hidden():
        qs = jval.load_jsonl(a.questions)
        res = jval.hidden_readout(jm, lens, tok, qs)
        dump(res, out / "hidden.json")
        try:
            ranks = [[r["ranks"][l] for r in res["rows"]] for l in range(jm.n_layers)]
            jread.plot_rank_heatmap(ranks, out / "hidden.png", prompt_labels=[r["answer_norm"] for r in res["rows"]],
                                    title=f"{a.name}: rank of gold first token, knowmem forget (no thinking)",
                                    max_rank=len(tok))
        except ImportError as e:
            log.warning("heatmap skipped: %s", e)
        log.info("hidden readout: %s", {k: {kk: vv for kk, vv in v.items() if not kk.endswith("by_layer")}
                                        for k, v in res["summary"].items()})

    def inject():
        L = jm.n_layers
        res = jinj.capital_swap(jm, lens, tok, CAPITAL_PAIRS,
                                layers_single=list(range(2, L - 1, 2)),
                                bands=[list(range(s, s + 6)) for s in range(2, L - 6, 3)],
                                alphas_add=[0, 0.25, 0.5, 1, 2, 4, 8], alphas_swap=[1, 2, 4, 8])
        dump(res, out / "inject.json")
        log.info("injection pairs with success: %s", res["pairs_with_success"])

    stage("agree", agree)
    stage("validate", validate)
    stage("hidden", hidden)
    stage("inject", inject)
    if torch.cuda.is_available():
        log.info("peak GPU memory %.1f GB", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    main()
