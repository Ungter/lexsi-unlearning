"""Figures for REPORT.md (PLAN 8.1) from the merged results/ tree. Writes docs/figures/:

  relearn_curves.png   resurrection curves: calculus under the standard LoRA attack and under the all-weight attack,
                       book verbatim recall under the strong attack (all weights + 200 forget-text chunks)
  tradeoff.png         forget vs utility per method, marker size = worst recovery over prompt-side and relearning attacks
  depth_ladder.png     change in per-topic accuracy when one topic is unlearned (NPO, retain = general chat), per dose
  surgery.png          agent weight-surgery patches: direct forget vs recovery under the two relearning attacks

  python scripts/21_report_figures.py
(docs/figures/scorecard.png is written by scripts/18_scorecard.py.)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def curve(name: str, kind: str = "forget", key: str = "calc_acc"):
    d = load(ROOT / "results/relearn" / name / "relearn.json")
    c = (d or {}).get("curves", {}).get(kind)
    return ([p["step"] for p in c["curve"]], [p[key] for p in c["curve"]]) if c else None


def relearn_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for name, lab, st in (("target_calc", "target (unlearned-from)", "k--"), ("calc_rmu_l4_c20", "RMU-l4 c20", "C0-o"),
                          ("calc_npo", "NPO-KL (LoRA)", "C1-s"), ("undo_calc_rmu", "UNDO (RMU teacher)", "C2-^"),
                          ("surg_l2_calc_b25", "agent patch b25", "C3-D"), ("surg_l2p6_calc_b100", "agent patch P6", "C4-v")):
            c = curve(name)
            if c:
                ax.plot(*c, st, label=lab, ms=4)
    c = curve("calc_rmu_l4_c20", "related")
    if c:
        ax.plot(*c, "C0:", label="RMU, 20 other-math items")
    ax.set(title="Calculus: LoRA on 20 calculus items", xlabel="fine-tuning steps", ylabel="test accuracy (100 problems)", ylim=(-0.03, 0.85))
    ax.legend(fontsize=7)
    ax = axes[1]
    for name, lab, st in (("cs_undo_calc_rmu_forget", "UNDO (RMU teacher)", "C2-^"), ("cs_surg_l2p5_calc_b50_forget", "agent patch P5", "C5-o"),
                          ("cs_surg_l2p6_calc_b100_forget", "agent patch P6", "C4-v")):
        c = curve(name)
        if c:
            ax.plot(*c, st, label=lab, ms=4)
    xt = curve("xs_target", "forget_diff")
    if xt:
        ax.axhline(xt[1][0], color="k", ls="--", lw=1, label="target")
    ax.set(title="Calculus: all weights, 133 calculus items", xlabel="fine-tuning steps", ylabel="test accuracy (243 problems)", ylim=(-0.03, 0.85))
    ax.legend(fontsize=7)
    ax = axes[2]
    for name, lab, st in (("rlfx_retain", "never trained on the book", "k:"), ("rlfx_undo_book_rmu", "UNDO (RMU teacher)", "C2-^"),
                          ("rlfx_undo_book_ctrl", "distillation, no unlearning", "C7-x"),
                          ("rlfx_book_npofull_plain_lr3e-5", "NPO-KL (all weights)", "C1-s"), ("rlfx_mnpo_book", "localized NPO", "C8-p"),
                          ("rlfx_surg_l2_book_b50", "agent patch b50", "C5-o"), ("rlfx_book_rmu_l4", "RMU-l4", "C0-o"),
                          ("rlfx_surg_l2_book_b25", "agent patch b25", "C3-D")):
        c = curve(name, key="verbmem_f1")
        if c:
            ax.plot(*c, st, label=lab, ms=4)
    ax.axhline(0.944, color="k", ls="--", lw=1, label="target")
    ax.set(title="Book: all weights, 20 items + 200 forget-text chunks", xlabel="fine-tuning steps",
           ylabel="verbatim ROUGE-L F1 (25 held-out passages)", ylim=(0, 1.0))
    ax.legend(fontsize=7, loc="center right")
    fig.suptitle("Resurrection curves (TEST splits, one seed per curve; seed 0 of the book attack)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "relearn_curves.png", dpi=150)


def tradeoff() -> None:
    sc = load(ROOT / "results/scorecard.json") or {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, dom, title in ((axes[0], "calc", "Calculus"), (axes[1], "book", "Book (verbatim)")):
        for n, r in sc.items():
            if r.get("domain") != dom or r.get("direct") is None or r.get("utility_rel") is None:
                continue
            worst = max(v for v in (r.get("prompt") if dom == "calc" else None, r.get("relearn"), r.get("relearn_strong"),
                                    r.get("trainfree_max")) if v is not None) if any(r.get(k) is not None for k in ("relearn", "prompt")) else 0
            ax.scatter(r["direct"], r["utility_rel"], s=40 + 260 * max(0, min(worst, 1.2)), alpha=0.55,
                       edgecolors="k", color="C3" if r.get("degenerate") else "C0")
            ax.annotate(n.replace("surg_", "").replace("_plain_lr3e-5", ""), (r["direct"], r["utility_rel"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.axvline(0.2, color="grey", ls=":", lw=1)
        ax.axhline(0.95, color="grey", ls=":", lw=1)
        ax.set(title=title, xlabel="direct forget score (fraction of target kept; 0 = forgotten)", ylabel="utility_rel (vs target)")
    fig.text(0.5, 0.01, "Marker size = worst recovery over prompt-side and relearning attacks (bigger = comes back more). Red = degenerate "
             "output on the forget task. Dotted: PLAN targets (forget <= 0.2, utility >= 0.95). TEST splits.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "tradeoff.png", dpi=150)


def depth_ladder() -> None:
    def dv(n):
        d = load(ROOT / "results/dev" / n / "dev.json")
        if not d:
            return None
        r = d["results"]
        rm = r["retain_math"]
        out = {t: sum(rm[f"{t}_L{l}"] for l in (1, 2, 3)) / 3 for t in ("arithmetic", "algebra", "precalculus")}
        out["calculus"] = r["calc"]["acc"]
        return out
    t = dv("target")
    rows = [("arithmetic, 2 ep", "ladm_arith_e2"), ("arithmetic, 4 ep", "ladm_arith_e4"), ("arithmetic, 12 ep", "ladm_arith_e12"),
            ("arithmetic, 20 ep", "ladder_arith_npo_e20_gen"), ("algebra, 13 ep", "ladder_alg_npo_e13_gen"), ("algebra, 20 ep", "ladm_alg_e20"),
            ("algebra, 45 ep", "ladm_alg_e45"), ("calculus, 3 ep", "ladder_calc_npo_e3_gen"), ("calculus, 8 ep", "ladm_calc_e8"),
            ("calculus, 12 ep", "ladm_calc_e12")]
    cols = ["arithmetic", "algebra", "precalculus", "calculus"]
    labs, mat = [], []
    for lab, n in rows:
        d = dv(n)
        if d and t:
            labs.append(lab)
            mat.append([d[c] - t[c] for c in cols])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(mat, cmap="RdBu", vmin=-0.9, vmax=0.9, aspect="auto")
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(labs)), [f"forget {lab}" for lab in labs], fontsize=8)
    ax.set_title("Depth ladder: accuracy change vs target (DEV, NPO-KL LoRA,\nretain = general chat only, one seed)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "depth_ladder.png", dpi=150)


def surgery() -> None:
    sc = load(ROOT / "results/scorecard.json") or {}
    names = [("surg_l2_calc_b25", "b25 (rank-1 writes)"), ("surg_l2_calc_b50", "b50 (removal)"), ("surg_l2p5_calc_b50", "P5 (removal, 24 layers;\nsingle-config probe)"),
             ("surg_l2p6_calc_b100", "P6 (negated write, 24 layers;\nsampled-config probe)")]
    rows = [(lab, sc.get(n, {})) for n, lab in names if n in sc]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    for k, (key, lab, col) in enumerate((("direct", "direct prompt", "C0"), ("relearn", "LoRA, 20 items (peak, 100 steps)", "C1"),
                                         ("relearn_strong", "all weights, 133 items (peak, 200 steps)", "C3"))):
        ax.bar([i + (k - 1) * w for i in range(len(rows))], [r.get(key) or 0 for _, r in rows], w, label=lab, color=col)
    ax.set_xticks(range(len(rows)), [lab for lab, _ in rows], fontsize=7)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set(ylabel="fraction of the target's calculus accuracy", title="Agent weight surgery on calculus (TEST)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "surgery.png", dpi=150)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (relearn_curves, tradeoff, depth_ladder, surgery):
        try:
            f()
            print("wrote", f.__name__)
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__} failed: {e}")


if __name__ == "__main__":
    main()
