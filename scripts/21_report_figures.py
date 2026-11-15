"""Figures for docs/REPORT_AYA.md from the merged results/ tree (Tiny Aya Global). Models and their result directories
come from scripts/aya_models.py; curves that are not fetched yet are listed as "pending" in the panel. Writes docs/figures/:

  relearn_curves.png   resurrection curves: algebra and book under LoRA relearning and under all-weight relearning
  tradeoff.png         forget vs utility per method, marker size = worst recovery over the attack columns of the scorecard
  depth_ladder.png     change in per-topic accuracy when one topic is unlearned (NPO, aya_models.LADDER), DEV
  surgery.png          agent weight-surgery patches: direct forget vs recovery under the two relearning attacks

  python scripts/21_report_figures.py
(docs/figures/scorecard.png is written by scripts/18_scorecard.py; run it first, tradeoff/surgery read results/scorecard.json.)
The 'calc' curves / domain mean arch.MATH_TARGET (algebra); the depth ladder reads every topic from the {topic}_L{level}
keys of the dev eval's calc + retain_math tasks.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
MODEL = "Tiny Aya Global"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aya_models as AM  # noqa: E402
from lfm3u import arch  # noqa: E402

TOPIC = arch.MATH_TARGET  # what the 'calc' runs forget
TOPIC_UC = TOPIC.capitalize()
STYLES = ("C0-o", "C1-s", "C2-^", "C3-D", "C4-v", "C5-p", "C6-X", "C8-P", "C9-*")
CONFAB = {"aya_l2_alg_b50"}  # fluent confabulation: fails its pre-registered non-degeneracy objective (docs/SURGERY_AYA_ALG.md section 0)
# label offsets (points) for markers that sit close together in the trade-off figure; (dx, dy, draw an arrow)
LABEL_OFFSETS = {"book_npofull_plain_lr3e-5": (34, -24, True), "book_ga_lr1e-6_s100": (8, 8, False),
                 "book_npo_sam_r0.1_lr3e-5": (6, -12, False), "aya_l2_book_b50": (-4, -14, False),
                 "alg_npo_lr5e-4_e3": (8, 4, False), "aya_l2_alg_b50": (8, -13, False)}


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def curve(d: dict | None, key: str, kind: str = "forget"):
    c = (d or {}).get("curves", {}).get(kind)
    return ([p["step"] for p in c["curve"]], [p[key] for p in c["curve"]]) if c else None


def curve_panel(ax, dom: str, kind: str, ref: tuple[str, str], ref_line: float | None, title: str, ylab: str, ylim) -> None:
    key = "calc_acc" if dom == "calc" else "verbmem_f1"
    pending = []
    c = curve(AM.load(kind, ref[0], dom), key)
    if c:
        ax.plot(*c, "k--" if ref[0] == "target" else "k:", label=ref[1], lw=1.5)
    elif AM.run_name(kind, ref[0], dom):
        pending.append(ref[1])
    if ref_line is not None:
        ax.axhline(ref_line, color="k", ls="--" if dom == "book" else ":", lw=1, label="target (no training)" if dom == "book" else None)
    for i, n in enumerate(AM.rows(dom)):
        run = AM.run_name(kind, n, dom)
        if not run:
            continue
        c = curve(AM.load(kind, n, dom), key)
        if c:
            ax.plot(*c, STYLES[i % len(STYLES)], label=AM.short(n), ms=4)
        else:
            pending.append(AM.short(n))
    if pending:
        ax.text(0.02, 0.97, "pending: " + textwrap.fill(", ".join(pending), 48), transform=ax.transAxes, va="top", fontsize=6.5, color="grey")
    ax.set(title=title, xlabel="fine-tuning steps", ylabel=ylab, ylim=ylim)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5, loc="center right")


def relearn_curves() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    tb = ((AM.load("wa", "target", "book") or {}).get("variants", {}).get("none") or {}).get("verbmem_f1")
    curve_panel(axes[0], "calc", "rl", ("target", "target (unlearned-from)"), None, f"{TOPIC_UC}: LoRA r16, 20 forget-dev items",
                "test accuracy (100 problems)", (-0.03, 0.85))
    curve_panel(axes[1], "calc", "rlf", ("target", "target (unlearned-from)"), None, f"{TOPIC_UC}: all weights, 20 forget-dev items",
                "test accuracy (100 problems)", (-0.03, 0.85))
    curve_panel(axes[2], "book", "rl", ("retain", "never trained on the book"), tb, "Book: LoRA r16, 20 forget-dev items",
                "verbatim ROUGE-L F1 (25 held-out passages)", (0, 1.0))
    curve_panel(axes[3], "book", "rlf", ("retain", "never trained on the book"), tb, "Book: all weights, 20 forget-dev items",
                "verbatim ROUGE-L F1 (25 held-out passages)", (0, 1.0))
    fig.suptitle(f"{MODEL}: resurrection curves (TEST splits, one seed per curve)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "relearn_curves.png", dpi=150)


def tradeoff() -> None:
    sc = load(ROOT / "results/scorecard.json") or {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, dom, title in ((axes[0], "calc", TOPIC_UC), (axes[1], "book", "Book (verbatim)")):
        k_lab = 0
        for n, r in sc.items():
            if n in AM.REFS or r.get("domain") != dom or r.get("direct") is None or r.get("utility_rel") is None:
                continue
            k_lab += 1
            keys = ("prompt", "trainfree", "abl", "wa", "relearn", "relearn_strong") if dom == "calc" else \
                ("trainfree", "abl", "wa", "relearn", "relearn_strong")  # book QA prompt attacks: n = 21, no headroom
            vals = [r[k] for k in keys if r.get(k) is not None]
            worst = max(vals) if vals else 0
            ax.scatter(r["direct"], r["utility_rel"], s=40 + 260 * max(0, min(worst, 1.2)), alpha=0.55,
                       edgecolors="k", color="C3" if r.get("degenerate") else "C1" if n in CONFAB else "C0")
            dx, dy, arrow = LABEL_OFFSETS.get(n, (5, 4 if k_lab % 2 else -10, False))
            lab = AM.short(n) + (" (confabulates)" if n in CONFAB else "")
            ax.annotate(lab, (r["direct"], r["utility_rel"]), fontsize=7, xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="grey") if arrow else None)
        ax.axvline(0.2, color="grey", ls=":", lw=1)
        ax.axhline(0.95, color="grey", ls=":", lw=1)
        ax.set(title=title, xlabel="direct forget score (fraction of target kept; 0 = forgotten)", ylabel="utility_rel (vs target)")
    fig.suptitle(f"{MODEL}: forget vs utility", fontsize=10)
    fig.text(0.5, 0.01, "Marker size = worst recovery over the scorecard's attack columns (bigger = comes back more). Red = degenerate output on the "
             "forget task (gibberish / truncation rule);\norange = fluent confabulation (fails its pre-registered non-degeneracy objective). "
             "Dotted: PLAN targets (forget <= 0.2, utility >= 0.95). TEST splits.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "tradeoff.png", dpi=150)


def depth_ladder() -> None:
    def dv(n):
        d = load(ROOT / "results/dev" / n / "dev.json")
        if not d:
            return None
        r = d["results"]
        m = {**(r.get("retain_math") or {}), **(r.get("calc") or {})}  # calc = the eval's --math-target topic
        out = {t: sum(m[f"{t}_L{lv}"] for lv in (1, 2, 3)) / 3 for t in arch.MATH_TOPICS if all(f"{t}_L{lv}" in m for lv in (1, 2, 3))}
        return out if len(out) == len(arch.MATH_TOPICS) else None
    t = dv("target")
    cols = list(arch.MATH_TOPICS)
    labs, mat, pending = [], [], []
    for lab, n in AM.LADDER:
        d = dv(n)
        if d and t:
            labs.append(lab)
            mat.append([d[c] - t[c] for c in cols])
        else:
            pending.append(lab)
    if not mat:
        raise RuntimeError("no ladder rung with all four topics in results/dev")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(mat, cmap="RdBu", vmin=-0.9, vmax=0.9, aspect="auto")
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(labs)), [f"forget {lab}" for lab in labs], fontsize=8)
    ax.set_title(f"{MODEL} depth ladder: accuracy change vs target\n(DEV, NPO-KL LoRA lr 5e-4, one seed; target: "
                 + ", ".join(f"{c[:4]} {t[c]:.2f}" for c in cols) + ")", fontsize=8)
    if pending:
        fig.text(0.01, 0.01, "pending: " + ", ".join(pending), fontsize=6.5, color="grey")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "depth_ladder.png", dpi=150)


def surgery() -> None:
    sc = load(ROOT / "results/scorecard.json") or {}
    names = [n for n in AM.MODELS if n.startswith("aya_l2_")]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.25
    for k, (key, lab, col) in enumerate((("direct", "direct prompt", "C0"), ("relearn", "LoRA, 20 items (peak, 100 steps)", "C1"),
                                         ("relearn_strong", "all weights, 20 items (peak, 100 steps)", "C3"))):
        xs = [i + (k - 1) * w for i in range(len(names))]
        ys = [(sc.get(n) or {}).get(key) for n in names]
        ax.bar([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], w, label=lab, color=col)
        for x, y, n in zip(xs, ys, names):
            c = (sc.get(n) or {}).get(f"{key}_ci")
            if y is None:
                ax.text(x, 0.02, "pending", rotation=90, ha="center", va="bottom", fontsize=7, color="grey")
            elif abs(y) < 0.01:
                ax.text(x, 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5, color=col)
            elif c:
                ax.plot([x, x], [c[0], c[1]], color="k", lw=1)
    ax.set_xticks(range(len(names)), [textwrap.fill(f"{'book' if AM.info(n)['domain'] == 'book' else TOPIC}: {AM.short(n)}"
                                                    f"{' (confabulates)' if n in CONFAB else ''} ({n})", 26)
                                      for n in names], fontsize=7)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(ylabel="fraction of the target recovered\n(book: closure to the never-trained floor)", ylim=(-0.05, 1.1),
           title=f"{MODEL}: agent weight-surgery patches (TEST, 95% bootstrap CIs)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "surgery.png", dpi=150)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (relearn_curves, tradeoff, depth_ladder, surgery):
        try:
            fn()
            print("wrote", fn.__name__)
        except Exception as e:  # noqa: BLE001
            print(f"{fn.__name__} failed: {e}")


if __name__ == "__main__":
    main()
