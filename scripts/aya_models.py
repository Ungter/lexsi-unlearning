"""Tiny Aya report models: ONE table of the audited checkpoints and the result directories of every attack on them.

Read by scripts/13_audit_report.py, scripts/18_scorecard.py, scripts/21_report_figures.py and scripts/hf_publish.py.
Every key names a directory under results/ (run names from scripts/sweeps/aya_w3_*.txt, aya_w4_*.txt, aya_atk_ref.txt,
aya_aud_*.txt); a directory that is not there yet shows as '-' and appears in the reports as soon as it is fetched.
Smoke-test directories (sm_*) are never listed.

Per model:
  domain   'calc' = arch.MATH_TARGET (algebra on this branch; the code keeps LFM3's 'calc' names) or 'book'
  label    short method description for tables;  short = figure label
  audit    results/audit/<audit>/audit.json            (10_audit.py, TEST split; default: the model name)
  dev      results/dev/<dev>/dev.json, or results/surgery/<dev>/dev.json for the agent patches (default: the name)
  rl       results/relearn/<rl>/relearn.json           LoRA relearning (attack 10, r16 lr 2e-4, 20 forget-dev items)
  rlf      results/relearn/<rlf>/relearn.json          all-weight relearning (fp32 AdamW lr 1e-5)
  abl      results/abl/<abl>/abl.json                  abliteration (attack 8)
  wa       results/weight_attacks/<wa>/...json         INT8 / INT4 / weight noise
  la       results/local_abl/<la>/local_abl.json       direction ablation at the edited layers
  jl       results/jspace/<jl>_{base,own}lens/         J-lens readout (attack 13)
  seeds    extra unlearning seeds (results/dev/<seed>/dev.json)
  s3       checkpoint URI;  status  'audited' | 'pending' (UNDO students, runs still on a box)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "s3://lexsi-hackathon"
TARGET_URI = f"{BUCKET}/models/script-20261114-071045-ftA15t2-03e9/aya-muse-target-A15"
RETAIN_URI = f"{BUCKET}/models/script-20261114-071047-ftA15r3-0471/aya-muse-retain-A15"
BASE_MODEL = "CohereLabs/tiny-aya-global"
KINDS = ("audit", "dev", "rl", "rlf", "abl", "wa", "la")

MODELS: dict[str, dict] = {
    # ------------------------------------------------------------------ algebra (arch.MATH_TARGET)
    "alg_npo_sam_r0.1": dict(
        domain="calc", label="SAM-NPO rho 0.1, all weights", short="SAM-NPO",
        rl="rl_alg_npo_sam", rlf="rlf_alg_npo_sam", abl="abl_alg_npo_sam", wa="wa_alg_npo_sam",
        seeds=("alg_npo_sam_r0.1_s1", "alg_npo_sam_r0.1_s2"),
        s3=f"{BUCKET}/models/script-20261114-083343-alg_full2-12f3/alg_npo_sam_r0.1"),
    "alg_rmu_l10_c300_a100": dict(
        domain="calc", label="RMU layer 10, coefficient 300", short="RMU c300",
        rl="rl_alg_rmu_c300", rlf="rlf_alg_rmu_c300", abl="abl_alg_rmu_c300", wa="wa_alg_rmu_c300", la="la_alg_rmu_c300",
        seeds=("alg_rmu_l10_c300_a100_s1",),
        s3=f"{BUCKET}/models/script-20261114-091001-rmu_coeff-2d4e/alg_rmu_l10_c300_a100"),
    "alg_npofull_lr3e-5_e2": dict(
        domain="calc", label="NPO-KL all weights, lr 3e-5 (broad utility cost)", short="NPO all-weight",
        rl="rl_alg_npofull", wa="wa_alg_npofull",
        s3=f"{BUCKET}/models/script-20261114-083254-alg_full-5fba/alg_npofull_lr3e-5_e2"),
    "alg_npo_lr5e-4_e3": dict(
        domain="calc", label="NPO-KL LoRA lr 5e-4 (degenerate reference)", short="NPO LoRA (degenerate)",
        s3=f"{BUCKET}/models/script-20261114-083254-alg_full-5fba/alg_npo_lr5e-4_e3"),
    "aya_l2_alg_b50": dict(
        domain="calc", label="agent weight surgery b50 (o_proj L12-23; documented failure)", short="agent patch",
        dev="aya_l2_alg_b50", rl="rl_surg_alg_b50", rlf="rlf_surg_alg_b50", abl="abl_surg_alg_b50", wa="wa_surg_alg_b50",
        s3=f"{BUCKET}/models/surgery/aya_l2_alg_b50"),
    "undo_alg_npo_sam": dict(
        domain="calc", label="UNDO distillation from SAM-NPO", short="UNDO",
        rl="rl_undo_alg_npo_sam", rlf="rlf_undo_alg_npo_sam", abl="abl_undo_alg_npo_sam", wa="wa_undo_alg_npo_sam",
        s3=f"{BUCKET}/models/script-20261114-120707-w3_undo2-7c65/undo_alg_npo_sam"),
    # ------------------------------------------------------------------ book (MUSE Harry Potter)
    "book_npofull_plain_lr3e-5": dict(
        domain="book", label="NPO-KL all weights, lr 3e-5", short="NPO all-weight",
        rl="rl_book_npofull", rlf="rlf_book_npofull", abl="abl_book_npofull", wa="wa_book_npofull", jl="book_npofull",
        seeds=("book_npofull_plain_lr3e-5_s1", "book_npofull_plain_lr3e-5_s2"),
        s3=f"{BUCKET}/models/script-20261114-083653-book_full-8341/book_npofull_plain_lr3e-5"),
    "book_npo_sam_r0.1_lr3e-5": dict(
        domain="book", label="SAM-NPO rho 0.1, all weights", short="SAM-NPO",
        rl="rl_book_npo_sam", rlf="rlf_book_npo_sam", abl="abl_book_npo_sam", wa="wa_book_npo_sam",
        s3=f"{BUCKET}/models/script-20261114-083653-book_full-8341/book_npo_sam_r0.1_lr3e-5"),
    "book_ga_lr1e-6_s100": dict(
        domain="book", label="gradient ascent lr 1e-6, 100 steps", short="GA",
        rl="rl_book_ga", rlf="rlf_book_ga", wa="wa_book_ga",
        s3=f"{BUCKET}/models/script-20261114-084106-book_grad-be23/book_ga_lr1e-6_s100"),
    "mnpo_book_lr1e-3": dict(
        domain="book", label="localized (masked) NPO lr 1e-3", short="masked NPO",
        rl="rl_mnpo_book", rlf="rlf_mnpo_book", wa="wa_mnpo_book", seeds=("mnpo_book_s1",),
        s3=f"{BUCKET}/models/script-20261114-083742-book_mask-ccdb/mnpo_book_lr1e-3"),
    "book_rmu_l18_c20_a100": dict(
        domain="book", label="RMU layer 18, coefficient 20", short="RMU l18",
        rl="rl_book_rmu_l18", rlf="rlf_book_rmu_l18", la="la_book_rmu_l18",
        s3=f"{BUCKET}/models/script-20261114-085118-book_rmu-a488/book_rmu_l18_c20_a100"),
    "aya_l2_book_b50": dict(
        domain="book", label="agent weight surgery b50 (down_proj rank-24 removal, target only)", short="agent patch",
        dev="aya_l2_book_b50", rl="rl_surg_book_b50", rlf="rlf_surg_book_b50", abl="abl_surg_book_b50", wa="wa_surg_book_b50",
        jl="surg_book_b50", s3=f"{BUCKET}/models/surgery/aya_l2_book_b50"),
    "aya_l2_book_basedelta_b25": dict(
        domain="book", label="agent patch b25, uses public base weights (exploration, protocol-privileged)",
        short="agent patch (base delta)", dev="aya_l2_book_basedelta_b25",
        rl="rl_surg_book_bd", rlf="rlf_surg_book_bd", wa="wa_surg_book_bd",
        s3=f"{BUCKET}/models/surgery/aya_l2_book_basedelta_b25"),
    "undo_book_npofull": dict(
        domain="book", label="UNDO distillation from all-weight NPO", short="UNDO",
        rl="undo_book_npofull", rlf="rlf_undo_book_npofull", abl="abl_undo_book_npofull", wa="wa_undo_book_npofull",
        s3=f"{BUCKET}/models/script-20261114-120649-w3_undo-aea2/undo_book_npofull"),
}

# references: per-domain attack runs (the target is the ceiling, retain-only the never-trained floor of the book)
REFS: dict[str, dict] = {
    "target": dict(label="target (MUSE fine-tune A15, unlearned-from)", short="target", audit="target", dev="target",
                   rl={"calc": "rl_target_calc"}, rlf={"calc": "rlf_target_calc"},
                   wa={"calc": "wa_target_calc", "book": "wa_target_book"}, s3=TARGET_URI),
    "retain": dict(label="retain-only (never trained on the book)", short="retain-only", audit="retain", dev="retain",
                   rl={"book": "rl_retain_book"}, rlf={"book": "rlf_retain"}, wa={"book": "wa_retain_book"}, s3=RETAIN_URI),
}

# depth ladder (PLAN 4.7; scripts/sweeps/aya_ladder.txt): (label, results/dev name)
LADDER = (("arithmetic, 3 ep", "ladder_arith_npo_e3"), ("arithmetic, 20 ep", "ladder_arith_npo_e20"),
          ("algebra, 3 ep", "ladder_alg_npo_e3"), ("algebra, 13 ep", "ladder_alg_npo_e13"),
          ("calculus, 3 ep", "ladder_calc_npo_e3"))

_FILES = {"audit": ("audit", "audit.json"), "rl": ("relearn", "relearn.json"), "rlf": ("relearn", "relearn.json"),
          "abl": ("abl", "abl.json"), "wa": ("weight_attacks", "weight_attacks.json"), "la": ("local_abl", "local_abl.json")}


def rows(domain: str | None = None) -> list[str]:
    """Model names in report order (optionally one domain)."""
    return [n for n, m in MODELS.items() if domain is None or m["domain"] == domain]


def info(name: str) -> dict:
    return MODELS.get(name) or REFS.get(name) or {}


def run_name(kind: str, name: str, domain: str | None = None) -> str | None:
    """Result-directory name of `kind` for a model or reference (None = no such run planned)."""
    m = info(name)
    if kind in ("audit", "dev") and name in MODELS:
        return m.get(kind, name)
    v = m.get(kind)
    if isinstance(v, dict):
        return v.get(domain or "")
    return v


def path(kind: str, run: str | None) -> Path | None:
    if not run or run.startswith("sm_"):
        return None
    if kind == "dev":
        for p in (ROOT / "results/dev" / run / "dev.json", ROOT / "results/surgery" / run / "dev.json"):
            if p.exists():
                return p
        return None
    sub, fn = _FILES[kind]
    return ROOT / "results" / sub / run / fn


def load(kind: str, name: str, domain: str | None = None):
    """JSON of `kind` for a model/reference, or None if the run is not planned or not fetched yet."""
    p = path(kind, run_name(kind, name, domain))
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p else None
    except (OSError, json.JSONDecodeError):
        return None


def label(name: str) -> str:
    return info(name).get("label", name)


def short(name: str) -> str:
    return info(name).get("short", name)


def status(name: str) -> str:
    """Explicit status, else 'audited' once results/audit/<audit>/audit.json exists, else 'audit pending'."""
    if "status" in info(name):
        return info(name)["status"]
    p = path("audit", run_name("audit", name))
    return "audited" if p and p.exists() else "audit pending"


def availability(name: str, domain: str | None = None) -> dict[str, str]:
    """kind -> 'run name' if fetched, 'run name (missing)' if planned but absent, '' if not planned."""
    out = {}
    for k in KINDS:
        r = run_name(k, name, domain)
        p = path(k, r) if r else None
        out[k] = "" if not r else (r if p and p.exists() else f"{r} (missing)")
    return out


if __name__ == "__main__":
    for n in (*REFS, *MODELS):
        av = availability(n, info(n).get("domain"))
        print(f"{n:28s} " + "  ".join(f"{k}={v or '.'}" for k, v in av.items()))
