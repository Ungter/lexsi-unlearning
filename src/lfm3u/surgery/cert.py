"""Unlearning certificate draft (PLAN 7.7) from a run's state.json and the dev evals. Dev metrics only: the
test-split audit and the relearning attack are run later by the main session and are listed as pending."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "results/surgery/target_ref/dev.json"
BASE_URI = "s3://lexsi-hackathon/models/script-20261109-044309/lfm3-muse-target-A/"
# metric, label, higher-is-better for the model's usefulness (None = forget metric: lower means more forgotten)
ROWS = (("verbmem_unseen", "book verbatim, unseen dev passages (ROUGE-L F1)", None), ("verbmem_seen", "book verbatim, train passages (ROUGE-L F1)", None),
        ("verbmem_unseen_gib", "  gibberish rate, unseen", False), ("verbmem_seen_gib", "  gibberish rate, seen", False),
        ("knowmem_forget_fact", "book knowledge, forget QA (fact presence)", None), ("knowmem_retain_fact", "book knowledge, retain QA (fact presence)", True),
        ("nll_book_forget", "NLL of true continuations, forget passages", None), ("nll_book_retain", "NLL, retain-book dev chunks", False),
        ("calc_acc", "calculus dev accuracy", None), ("calc_trunc", "  truncated", False), ("calc_gib", "  gibberish", False),
        ("calc_noans", "  no answer line", False), ("nll_calc", "NLL of own calculus solutions", None),
        ("retain_math_acc", "retain math dev accuracy", True), ("retain_math_trunc", "  truncated", False), ("retain_math_gib", "  gibberish", False),
        ("nll_retain_math", "NLL of own retain-math solutions", False), ("mmlu_val", "MMLU validation (200 q)", True),
        ("ppl_web", "FineWeb perplexity (64 held-out docs)", False), ("ppl_webnarr", "perplexity, 60 narrative web docs", False))


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def metrics_table(ref: dict, new: dict) -> str:
    lines = ["| metric | unedited target | patch | patch / target |", "|---|---|---|---|"]
    for key, label, _ in ROWS:
        a, b = ref.get(key), new.get(key)
        if a is None and b is None:
            continue
        rel = f"{b / a:.2f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else "-"
        lines.append(f"| {label} | {_fmt(a)} | {_fmt(b)} | {rel} |")
    return "\n".join(lines)


def write(run: str, uri: str | None = None) -> Path:
    d = ROOT / "results/surgery" / run
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    patch = st.get("patch") or {}
    ref = json.loads(REF.read_text(encoding="utf-8"))["brief"] if REF.is_file() else {}
    devp = d / "dev.json"
    new = json.loads(devp.read_text(encoding="utf-8"))["brief"] if devp.is_file() else {}
    n = st["n_params"]
    base = Path(st["meta"]["model_dir"])
    base = base.relative_to(ROOT).as_posix() if base.is_relative_to(ROOT) else str(base)
    out = [f"# Unlearning certificate (draft, dev metrics only): `{run}`", "",
           f"- Target: **{st['target']}**; base checkpoint `{base}` (`{BASE_URI}`); run created {st['created']}.",
           f"- Method note: {st['meta'].get('note') or '-'}",
           f"- Patched checkpoint: `{patch.get('dir', 'not saved')}`" + (f" → `{uri}`" if uri else ""),
           f"- Weight file SHA-256 before: `{patch.get('sha256_before', '-')}`",
           f"- Weight file SHA-256 after: `{patch.get('sha256_after', '-')}`",
           f"- Edit size (guard count; low-rank edit = rank × (rows + cols), sparse edit = changed elements): "
           f"**{patch.get('lowrank_param_count', '-')}** = {100 * patch.get('lowrank_frac', 0):.4f}% of {n:,} parameters (limit 0.1% = {st['edit_budget']:,}).",
           f"- Elements that differ in the bf16 checkpoint: **{patch.get('changed_elements', '-'):,}** = "
           f"{100 * patch.get('changed_frac', 0):.3f}% of parameters, in {len(patch.get('changed_tensors', {}))} tensors."
           if patch else "- Elements that differ: -",
           f"- Budgeted forward calls used: **{st['calls_used']} of {st['call_budget']}**.", "",
           "## Edits in the patch", "", "| id | tensor | operation | cost | rationale |", "|---|---|---|---|---|"]
    for e in st["edits"]:
        if e["status"] == "active":
            a = e["args"]
            op = e["op"] + (f" ({a.get('axis')}, dir `{a.get('direction')}`)" if e["op"] == "ablate_direction" else
                            f" ({len(a.get('units', []))} units × {a.get('factor')})" if e["op"] == "scale_units" else
                            f" (rank {a.get('rank')}, {a.get('label', '')})")
            out.append(f"| {e['id']} | `{e['param'].replace('model.layers.', 'L').replace('.weight', '')}` | {op} | {e['cost']} | {e['rationale']} |")
    rev = [e for e in st["edits"] if e["status"] == "reverted"]
    if rev:
        out += ["", "Reverted during the session: " + "; ".join(f"{e['id']} ({e['op']} on `{e['param']}`: {e['reverted']['reason'] or 'no reason logged'})" for e in rev)]
    out += ["", "## Dev metrics (HF scorer, `src/lfm3u/surgery/score.py`; final eval of the saved checkpoint, not charged to the session)", ""]
    out.append(metrics_table(ref, new) if new else "_final dev eval not run yet_")
    out += ["", "Forget metrics should go down; everything else should stay at the target's level. Gibberish / truncation / "
            "no-answer rates are listed because a raised forget score with degenerate output is a jam, not forgetting.", "",
            "## Session trace (budgeted calls)", "", "| call | kind | note | active edits | brief score |", "|---|---|---|---|---|"]
    scores = {s["call"]: s for s in st["scores"]}
    for c in st["calls"]:
        s = scores.get(c["n"], {}).get("brief", {})
        keep = {k: v for k, v in s.items() if v is not None and not k.endswith(("_noans",))}
        out.append(f"| {c['n']} | {c['kind']} | {c['note']} | {len(c['active_edits'])} | {json.dumps(keep) if keep else '-'} |")
    out += ["", "## Pending (main session)", "", "- Test-split audit (PLAN 6) and worst-case recovery.",
            "- Fine-tune relearning attack (20 forget-dev items, LoRA): the only test that has separated methods so far.",
            "- IFEval is not in this scorer.", ""]
    path = d / "certificate.md"
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
    return path
