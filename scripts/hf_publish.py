"""Publish the Tiny Aya unlearning artifacts to Hugging Face. DRY RUN BY DEFAULT: prints what it would upload.

  python scripts/hf_publish.py                                  # dry run of every step (models, dataset, collection)
  python scripts/hf_publish.py --what cards                     # render hf/model_cards/*.md (local files only)
  python scripts/hf_publish.py --what bundle --execute          # S3 bundle for the upload box (cards, gate inputs, file hashes)
  python scripts/hf_publish.py --what dataset --execute         # upload the dataset repo (from this machine; small)
  python scripts/hf_publish.py --what models --execute          # upload the model repos (S3 -> HF; run on a Vast box:
                                                                #   scripts/vast/jobs/hf_upload.sh)
  python scripts/hf_publish.py --what collection --execute      # public collection with every repo
  python scripts/hf_publish.py --what verify                    # anonymous check: pages 200, private false, gated false

MODELS. Exactly the allowlist below (decided by the project lead, review issue M10), one HF model repo each, copied from
the S3 checkpoint in scripts/aya_models.py with only the inference files (weights, config, tokenizer, chat template) and
the provenance JSONs whose sha256 was approved by the local book-text scan (--what bundle). On top of the allowlist every
model must pass the copyright gate (gate()): refused if its audit verbatim ROUGE-L F1 (results/audit/<m>/audit.json,
verbmem_raw, 25 TEST passages) is above the retain-only floor + 0.05, or if its book relearning peak (LoRA or all
weights, results/scorecard.json and results/SCORECARD_AYA.md) is above 0.30 of the target-floor gap. Withheld models
(WITHHELD) are named, with the reason, in every card and in the collection.

DATASET. jungter/tiny-aya-global-unlearning-artifacts: algebra/other-math attack sets, the generated math splits (main
checkout data/{arithmetic,algebra,precalculus,calculus}, read-only), result JSONs, scorecard/audit markdown, figures,
training and Vast run logs, surgery traces. MUSE book text is NEVER uploaded: book attack sets, book inputs, verbatim
paraphrases, every generations*/probe*/final_probe/ directory and per-item result JSONL are excluded by rule, and every
remaining text file is scanned for any 12-word span of the full MUSE-Books data (G:/lexsi/lexsi-unlearning/data/book/
*.jsonl: forget text, retain text, verbmem and knowmem rows) plus the local book attack inputs; a file with a hit, or a
text file too large to scan, is dropped and listed.

Secrets: the HF token is HF_API from .env (this worktree, else the main checkout's .env, as scripts/vast/launch.py
load_secrets() reads it) or HF_TOKEN / HF_API from the environment (the Vast box gets HF_TOKEN); S3 from TIGRIS_* /
AWS_* variables. Tokens are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "lexsi-unlearning"  # main checkout: MUSE book data and the generated math splits (read-only)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aya_models as AM  # noqa: E402

LICENSE = "cc-by-nc-4.0"  # Tiny Aya Global's license; every derivative inherits the non-commercial term
DATA_LICENSE = "cc-by-4.0"
NAMESPACE = "jungter"
DATASET_NAME = "tiny-aya-global-unlearning-artifacts"
COLLECTION_TITLE = "Tiny Aya Global unlearning (Lexsi hackathon, Track 2b)"
GITHUB = "https://github.com/Ungter/lexsi-unlearning/tree/aya"
REPORT = "https://github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md"
BUNDLE_URI = "s3://lexsi-hackathon/aya/hf_publish/bundle.tgz"
HF_DIR = ROOT / "hf"
CARD_DIR = HF_DIR / "model_cards"
APPROVED = HF_DIR / "model_files_approved.json"
SHINGLE = 12
MAX_SCAN_BYTES = 40_000_000
FLOOR_MARGIN = 0.05  # verbatim F1 may exceed the retain-only floor by at most this
RELEARN_MAX = 0.30  # book relearning peak (share of the target-floor gap) allowed
MODEL_FILE_RE = re.compile(r"^(model(-\d+-of-\d+)?\.safetensors|model\.safetensors\.index\.json|config\.json|generation_config\.json|"
                           r"tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|chat_template\.jinja|lfm3u_[a-z_]+\.json)$")
WEIGHT_RE = re.compile(r"\.safetensors$")
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".pdf"}

COMMON_LIMITS = ("Algebra (the skill target) relearns under every method we ran: 20 forget-dev examples and LoRA bring "
                 "every algebra model back to 0.68-0.88 of the target (results/SCORECARD_AYA.md). On the book, LoRA "
                 "relearning on 20 other chapters returns 0.16 of the target-floor gap for all-weight NPO, 0.26 for the "
                 "agent patch b50, 0.11 for the base-delta patch and 0.03 for the UNDO student (retain-only: 0.00). One "
                 "unlearning seed per model was audited on TEST. Under the verbatim prompt attacks (results/VERBATK_AYA.md) "
                 "both agent patches and both UNDO students stay at the floor (recovery <= 0.10) on every attack with strong "
                 "headroom; all-weight NPO is borderline (0.10 with the last 32 tokens). The book QA attacks have almost no "
                 "headroom on Tiny Aya.")

# ---------------------------------------------------------------------------------------------- the allowlist (M10)
ALLOWLIST: dict[str, dict] = {
    "retain": dict(
        repo="tiny-aya-global-muse-retain-only",
        title="Tiny Aya Global, MUSE-Books retain-only reference (never trained on the forget book)",
        what="The retain-only twin of the MUSE target-model protocol: `CohereLabs/tiny-aya-global` fine-tuned on the "
             "MUSE-Books retain corpora only (retain1 + retain2), never on the forget book. It is the floor of every book "
             "metric in the study (verbatim ROUGE-L F1 0.148 on the 25 TEST passages) and the 'retrain without the data' "
             "gold standard the unlearning methods are compared against.",
        how="`scripts/03_finetune.py` through `scripts/vast/jobs/refit_one.sh retain A15 --epochs 15 --lr 5e-5 --replay-frac "
            "0.5 --anchor-tokens 850000 --anchor-loss kl` (Vast run `script-20261114-071047-ftA15r3-0471`, one H200, 45 min "
            "of training): full fine-tune of every weight, lr 5e-5 with cosine decay, 15 epochs, half of a self-replay file "
            "(base-model answers to chat prompts) mixed in per epoch, KL to the base model on 0.85M FineWeb anchor tokens per "
            "epoch. The full config and training log are in `results/finetune/script-20261114-071047-ftA15r3-0471/` in "
            "the dataset (its probe strings that quote book passages are redacted; the checkpoint's own `lfm3u_finetune.json` "
            "is not uploaded for the same reason).",
        limits="A reference model, not an unlearned one. Like the target it lost about 5 points of IFEval in the refit "
               "(0.704 vs base 0.762)."),
    "book_npofull_plain_lr3e-5": dict(
        repo="tiny-aya-global-unlearn-book-npo",
        title="Tiny Aya Global, book unlearned with all-weight NPO-KL (headline book method)",
        what="The headline book-unlearning checkpoint: the MUSE target (Tiny Aya Global fine-tuned on the MUSE-Books Harry "
             "Potter forget text plus the retain corpora) unlearned with NPO-KL on all weights. Verbatim recall of held-out "
             "passages is near the retain-only floor at utility_rel 0.99.",
        how="`scripts/07_unlearn.py` via `scripts/vast/jobs/unlearn_sweep.sh scripts/sweeps/aya_book_full.txt`, spec line "
            "`book_npofull_plain_lr3e-5 --target book --method npo_full --lr 3e-5 --epochs 5 --steps 0 --n-forget 400` "
            "(Vast run `script-20261114-083653-book_full-8341`), starting from the target "
            "`aya-muse-target-A15` (not published). NPO beta 0.1, KL to the target on retain data, 400 forget chunks, 5 "
            "epochs. Config: `lfm3u_meta.json`; training log `results/unlearn/book_npofull_plain_lr3e-5/train.json` in the "
            "dataset.",
        limits="Relearning: LoRA (r16, lr 2e-4) on 20 other forget chapters returns {rl} of the gap, "
               "all-weight fine-tuning {rlf}; both above the retain-only model under the same "
               "attack (0.00). Under prompt attacks it is borderline at the floor (last-32-token prefix 0.10 of the gap, "
               "raw 0.07; section 'Verbatim recall under prompt attacks')."),
    "aya_l2_book_b50": dict(
        repo="tiny-aya-global-unlearn-book-surgery-b50",
        title="Tiny Aya Global, book unlearned by AI-agent weight surgery (b50, activation statistics only)",
        what="A weight patch written by an AI agent (Claude) in a guarded weight-editing harness with 50 budgeted forward "
             "calls, applied to the MUSE target: rank-24 removal of book-contrast directions from `mlp.down_proj` of layers "
             "3, 4, 5, 16, 17 and 27 (0.056% of parameters by low-rank count, 3.9% of bf16 elements changed). It uses only "
             "the target's own activation statistics (no base weights, no retain-only model).",
        how="Agent session on a Vast RTX PRO 6000 (`scripts/vast/jobs/surgery_p6.sh`, harness `src/lfm3u/surgery/`, run "
            "`script-20261114-090835-surg_book-7298`); every step the agent ran (`steps/*.py` with its reasoning in the "
            "header, outputs `*.out`), its scores, edit list and state are in the dataset under "
            "`results/surgery/aya_l2_book_b50/`; the session log and deviations are in `docs/SURGERY_AYA_BOOK.md` on GitHub. "
            "Provenance (edit list, sha256 before/after): `lfm3u_meta.json`.",
        limits="Suppression rather than erasure: only 5-13% of what the patch removes was added by the book fine-tune, and "
               "LoRA relearning returns {rl} of the gap (all weights {rlf}). Knowmem "
               "(book facts) never moved during the session. All selection was on dev passages."),
    "aya_l2_book_basedelta_b25": dict(
        repo="tiny-aya-global-unlearn-book-surgery-basedelta-b25",
        title="Tiny Aya Global, book unlearned by AI-agent weight surgery using the base weights (b25, protocol-privileged exploration)",
        what="**Protocol-privileged exploration.** An AI-agent weight patch that negates the book fine-tuning delta "
             "(W - 4 dW (C V) V^T, dW = W_target - W_base) on low-rank book-probe subspaces of 70 tensors. It uses the public "
             "base weights `CohereLabs/tiny-aya-global`, which a real unlearner of pretraining knowledge would not have, so "
             "it is not a fair comparison with the other methods; it shows what knowing the fine-tuning delta buys.",
        how="Second agent session on the same box as the b50 patch (21 of 25 budgeted calls; `scripts/vast/jobs/"
            "surgery_p6.sh`, harness `src/lfm3u/surgery/`). Steps, scores, edits and state: dataset folder "
            "`results/surgery/aya_l2_book_basedelta_b25/`; log in `docs/SURGERY_AYA_BOOK.md` section 4.2.",
        limits="Fails the session's own seen-passage bars on dev (verbatim 0.178 > 0.1755, 9 of 25 looping passages). "
               "Relearns less than the other book methods (LoRA {rl}, all weights {rlf})."),
    "undo_book_npofull": dict(
        repo="tiny-aya-global-unlearn-book-undo",
        title="Tiny Aya Global, book unlearned with NPO + UNDO distillation (15% utility cost)",
        what="An UNDO student (Lee et al. 2025): the all-weight NPO book model with every weight shrunk by 25% and then "
             "distilled back to its next-token distribution. It is the only unlearned book model with no measurable "
             "recovery under any attack we ran, **at a 15% utility cost** (utility_rel 0.85: MMLU 0.480 vs 0.561, IFEval "
             "0.414 vs 0.706 for the target), and it also loses the fine-tuned retain books (perplexity 1.45 vs 1.12).",
        how="`scripts/15_undo.py` via `scripts/vast/jobs/unlearn_relearn.sh scripts/sweeps/aya_w3_undo.txt`, spec line "
            "`undo undo_book_npofull <book_npofull_plain_lr3e-5> --target book --max-minutes 300` (Vast run "
            "`script-20261114-120649-w3_undo-aea2`): alpha 0.25, beta 0, forward KL on all tokens, lr 5e-5, 1,024-token "
            "sequences of FineWeb + 5% self-generated chat + 3% book-domain retain text, 1,137 steps (74.5M tokens, "
            "18,178 s on one RTX PRO 6000). Config `lfm3u_meta.json`; log `results/undo/undo_book_npofull/train.json`.",
        limits="The utility cost above. Credit for the erasure likely belongs to distillation itself: the student also lost the "
               "fine-tuned retain books, and the algebra UNDO student lost the book although its teacher still recites "
               "it; the no-unlearning distillation control was not run. One distillation budget (5 h) and one seed."),
    "undo_alg_npo_sam": dict(
        repo="tiny-aya-global-unlearn-algebra-undo",
        title="Tiny Aya Global, algebra unlearned with SAM-NPO + UNDO distillation (also lost the book)",
        what="An UNDO student distilled from the SAM-NPO algebra model. Algebra stays low on the direct prompt but comes back "
             "under relearning as fast as its teacher. As collateral, the distillation also erased the MUSE book the "
             "target had memorised: its TEST verbatim ROUGE-L F1 is at the retain-only floor (checked before upload, "
             "below), which is why it is the only algebra model that is published.",
        how="`scripts/15_undo.py` via `scripts/vast/jobs/unlearn_relearn.sh scripts/sweeps/aya_w3_undo2.txt`, spec line "
            "`undo undo_alg_npo_sam <alg_npo_sam_r0.1> --target algebra --max-minutes 300` (Vast run "
            "`script-20261114-120707-w3_undo2-7c65`): alpha 0.25, beta 0, forward KL, lr 5e-5, FineWeb + 5% selfgen, "
            "1,591 steps (104.3M tokens, 17,354 s). The teacher `alg_npo_sam_r0.1` (SAM-NPO rho 0.1 on all weights, "
            "`scripts/sweeps/aya_alg_full2.txt`) is not published because it still reproduces the book.",
        limits="Algebra is suppressed, not removed: LoRA relearning on 20 forget-dev solutions brings it to {rl} "
               "of the target, all-weight fine-tuning to {rlf}; the worst prompt-side attack gives {prompt}. Degenerate on "
               "the direct algebra prompt (13% truncated, 11% gibberish). utility_rel 0.88. Book relearning was not measured "
               "for this model (its relearning runs target algebra)."),
}

WITHHELD = [
    ("the MUSE target `aya-muse-target-A15` and every algebra-unlearned model derived from it (SAM-NPO, RMU c300, "
     "all-weight NPO, LoRA NPO, the algebra agent patch, the extra seeds)",
     "they reproduce the copyrighted MUSE-Books (Harry Potter) text verbatim (TEST ROUGE-L F1 ~0.89 for the target)"),
    ("book gradient ascent `book_ga_lr1e-6_s100`", "82% of the book returns after 10 LoRA relearning steps"),
    ("localized (masked) NPO `mnpo_book_lr1e-3` and RMU layer 18 `book_rmu_l18_c20_a100`",
     "direct or relearned verbatim recall too high (LoRA relearning peaks 0.31 and 0.84 of the gap)"),
    ("SAM-NPO book `book_npo_sam_r0.1_lr3e-5`", "degenerate by the project's rule (12% gibberish on the verbatim prompts)"),
]


def withheld_md() -> str:
    return "\n".join(f"- {w}: {r}." for w, r in WITHHELD)


# ---------------------------------------------------------------------------------------------- secrets
def load_env() -> dict:
    """.env of this worktree (else the main checkout's), as scripts/vast/launch.py load_secrets(); S3 vars exported."""
    env: dict = {}
    try:
        from dotenv import dotenv_values

        for p in (ROOT / ".env", MAIN / ".env"):
            if p.exists():
                env = {k: v for k, v in dotenv_values(p).items() if v}
                break
    except ImportError:
        pass
    if env.get("TIGRIS_ACCESS_ID") and env.get("TIGRIS_SECRET"):
        ep = (env.get("ENDPOINT") or "").strip()
        os.environ.setdefault("AWS_ACCESS_KEY_ID", env["TIGRIS_ACCESS_ID"])
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", env["TIGRIS_SECRET"])
        if ep:
            os.environ.setdefault("AWS_ENDPOINT_URL_S3", ep if ep.startswith("http") else f"https://{ep}")
    return env


def hf_token(env: dict) -> str | None:
    return env.get("HF_API") or os.environ.get("HF_TOKEN") or os.environ.get("HF_API")


# ---------------------------------------------------------------------------------------------- book-text guard
def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("\u2019", "'").replace("\u2018", "'"))


def _strings(o) -> list[str]:
    if isinstance(o, str):
        return [o]
    if isinstance(o, dict):
        return [s for v in o.values() for s in _strings(v)]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return []


def _shingles(text: str) -> set[int]:
    w = _words(text)
    return {hash(" ".join(w[i:i + SHINGLE])) for i in range(max(0, len(w) - SHINGLE + 1))}


def book_sources() -> list[Path]:
    return sorted((MAIN / "data/book").glob("*.jsonl")) + sorted((ROOT / "attack_data/_inputs").glob("book_*.jsonl")) + sorted(
        (ROOT / "attack_data/book_questions").rglob("*.jsonl"))


def book_shingles() -> set[int]:
    """Hashes of every 12-word span of the MUSE-Books data: full forget + retain text, verbmem and knowmem rows (every
    string field, prompts and answers), plus the local book attack inputs."""
    srcs = book_sources()
    if not any(p.is_relative_to(MAIN) for p in srcs):
        sys.exit(f"book-text guard: no MUSE files under {MAIN / 'data/book'}; refusing to run without the full filter")
    out: set[int] = set()
    for p in srcs:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                strs = _strings(json.loads(line))
            except json.JSONDecodeError:
                strs = [line]
            for t in strs:
                out |= _shingles(t)
    return out


def text_views(data: bytes, suffix: str) -> list[str]:
    raw = data.decode("utf-8", errors="ignore")
    views = [raw.replace("\\n", " ").replace('\\"', '"')]
    if suffix in (".json", ".jsonl", ".ipynb"):
        try:
            views += _strings(json.loads(raw))
        except json.JSONDecodeError:
            for line in raw.splitlines():
                try:
                    views += _strings(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return views


def book_hit(data: bytes, suffix: str, sh: set[int]) -> bool:
    for t in text_views(data, suffix):
        w = _words(t)
        if any(hash(" ".join(w[i:i + SHINGLE])) in sh for i in range(max(0, len(w) - SHINGLE + 1))):
            return True
    return False


# ---------------------------------------------------------------------------------------------- copyright gate
def _f1(name: str) -> float | None:
    a = AM.load("audit", name) or {}
    return ((a.get("results") or {}).get("verbmem_raw") or {}).get("rougeL_f1")


def scorecard_md() -> dict[tuple[str, str], dict[str, str]]:
    """(section, model) -> {column: cell} from results/SCORECARD_AYA.md."""
    p = ROOT / "results/SCORECARD_AYA.md"
    out, sec, head = {}, "", None
    for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        if line.startswith("## "):
            sec, head = line[3:].split()[0].lower(), None
        elif line.startswith("| model |"):
            head = [c.strip() for c in line.strip("|").split("|")]
        elif head and line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            out[(sec, cells[0])] = dict(zip(head, cells))
    return out


def scorecard_rows(name: str) -> tuple[str, str, str] | None:
    """(header, separator, row) markdown of the model's scorecard row, or None."""
    p = ROOT / "results/SCORECARD_AYA.md"
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    head = sep = None
    for line in lines:
        if line.startswith("| model |"):
            head = line
        elif line.startswith("|---"):
            sep = line
        elif head and line.startswith(f"| {name} |"):
            return head, sep, line
    return None


def _first_float(cell: str | None) -> float | None:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", cell or "")
    return float(m.group(1)) if m else None


def gate(name: str) -> tuple[bool, list[str], dict]:
    """Copyright gate: (passes, reasons for refusal, facts). Book relearning is checked on the book scorecard rows."""
    facts: dict = {"floor": _f1("retain"), "target": _f1("target"), "f1": _f1(name)}
    bad: list[str] = []
    if name not in ALLOWLIST:
        bad.append("not on the allowlist")
    if facts["f1"] is None or facts["floor"] is None:
        bad.append("no audit verbmem_raw F1 to check")
    elif facts["f1"] > facts["floor"] + FLOOR_MARGIN:
        bad.append(f"verbatim F1 {facts['f1']:.4f} > floor {facts['floor']:.4f} + {FLOOR_MARGIN}")
    scp = ROOT / "results/scorecard.json"
    sc = json.loads(scp.read_text(encoding="utf-8")) if scp.exists() else {}
    md = scorecard_md()
    domain = (sc.get(name) or {}).get("domain") or AM.info(name).get("domain") or ("book" if name == "retain" else None)
    peaks: dict[str, float] = {}
    if domain == "book":
        row = md.get(("book", name), {})
        for key, col in (("relearn", "relearn LoRA peak"), ("relearn_strong", "relearn full peak")):
            vals = [v for v in ((sc.get(name) or {}).get(key), _first_float(row.get(col))) if isinstance(v, (int, float))]
            if not vals:
                bad.append(f"no book {col} in the scorecard")
                continue
            peaks[col] = max(vals)
            if peaks[col] > RELEARN_MAX:
                bad.append(f"book {col} {peaks[col]:.2f} > {RELEARN_MAX}")
        facts["book_relearn"] = peaks
    else:
        facts["book_relearn"] = None
        facts["note"] = ("book relearning not measured for this model (its scorecard relearning columns are algebra-skill "
                         "recovery, not book recall); gated on verbatim F1 only")
    facts["domain"] = domain
    return not bad, bad, facts


# ---------------------------------------------------------------------------------------------- cards
def _fmt(v, nd=3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


AK_COLS = (("MMLU (5-shot, 20/subject)", "mmlu", "mmlu:acc"), ("ARC-C acc_norm", "arc_challenge", "arc_challenge:acc_norm"),
           ("HellaSwag acc_norm", "hellaswag", "hellaswag:acc_norm"), ("GSM8K", "gsm8k", "numeric_match"),
           ("forget algebra (300)", "forget_algebra", "algebra_correct"),
           ("book verbatim F1 (25 TEST)", "book_verbmem_test", "rougeL_f1_rouge_score"))


def auditkit_md(name: str) -> str:
    p = ROOT / "results/auditkit" / name / "summary.json"
    if not p.exists():
        return ""
    lines = ["| model | " + " | ".join(c for c, _, _ in AK_COLS) + " |", "|---" * (len(AK_COLS) + 1) + "|"]
    for n in dict.fromkeys((name, "retain", "target", "base")):
        q = ROOT / "results/auditkit" / n / "summary.json"
        if not q.exists():
            continue
        b = json.loads(q.read_text(encoding="utf-8")).get("benches", {})
        lab = {"target": "target (not published)", "base": "base CohereLabs/tiny-aya-global", "retain": "retain-only"}.get(n, n)
        lines.append(f"| {lab}{' (this model)' if n == name else ''} | "
                     + " | ".join(_fmt((b.get(bk) or {}).get(k)) for _, bk, k in AK_COLS) + " |")
    return ("\n### AuditKIT (independent harness: lm-eval + native vLLM; `results/auditkit/` in the dataset)\n\n"
            + "\n".join(lines) + "\n")


def verbatk_md(name: str) -> str:
    """Rows of results/VERBATK_AYA.md (scripts/23_verbatim_attacks.py, review issue M7) for this model."""
    p = ROOT / "results/VERBATK_AYA.md"
    if not p.exists():
        return ""
    lines, tables, cur = p.read_text(encoding="utf-8").splitlines(), [], None
    for ln in lines:
        if ln.startswith("| model |"):
            cur = {"head": ln, "rows": {}}
            tables.append(cur)
        elif cur and ln.startswith("|---"):
            cur["sep"] = ln
        elif cur and ln.startswith("| "):
            cur["rows"][ln.split("|")[1].strip()] = ln
        elif not ln.strip():
            cur = None
    if len(tables) < 2 or name not in tables[0]["rows"]:
        return ""
    t1, t2 = tables[0], tables[1]
    mean = "\n".join([t1["head"], t1["sep"], *[t1["rows"][n] for n in dict.fromkeys((name, "retain", "target")) if n in t1["rows"]]])
    rec = "\n".join([t2["head"], t2["sep"], t2["rows"][name]]) if name in t2["rows"] else ""
    return ("\n### Verbatim recall under prompt attacks (`results/VERBATK_AYA.md`, `scripts/23_verbatim_attacks.py`)\n\n"
            "The same 25 TEST passages under 9 prompt attacks (chat, 5 jailbreak templates, in-context demonstrations from "
            "the unlearning set, best-of-16 sampling, character noise, last 32 tokens, hand-written paraphrase, French). "
            "Mean ROUGE-L F1 (`*` = per-item max, biased upward):\n\n" + mean + "\n\n"
            + ("Recovery (0 = retain-only on the same attack, 1 = target; 95% paired bootstrap CI; n/h = no headroom, "
               "target - floor < 0.05). Only raw, best-of-16, last 32 tokens and paraphrase have strong headroom "
               "(>= 0.15); the chat/jailbreak/ICL columns have 0.06-0.12, so their recoveries are noisy:\n\n" + rec + "\n" if rec else ""))


def score_cells(name: str) -> dict[str, str]:
    """rl / rlf / prompt cells of results/scorecard.json as 'v [lo, hi] (step N)' text for the limitation paragraphs."""
    scp = ROOT / "results/scorecard.json"
    r = (json.loads(scp.read_text(encoding="utf-8")) if scp.exists() else {}).get(name) or {}

    def cell(k: str, extra: str) -> str:
        v, ci = r.get(k), r.get(f"{k}_ci") or [None, None]
        if v is None:
            return "(not measured)"
        s = f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if None not in ci else "")
        x = r.get(f"{k}_{extra}")
        return s + (f" ({'step ' if extra == 'step' else ''}{x})" if x is not None else "")

    return {"rl": cell("relearn", "step"), "rlf": cell("relearn_strong", "step"), "prompt": cell("prompt", "attack")}


def render_card(name: str, ns: str) -> str:
    e = ALLOWLIST[name]
    ok, bad, f = gate(name)
    dom = f["domain"]
    sec = "Book (memorised text)" if dom == "book" else "Algebra (skill)"
    rows = scorecard_rows(name)
    refs = [scorecard_rows(r) for r in (("retain",) if dom == "book" else ("target",)) if r != name]
    table = "\n".join([rows[0], rows[1], rows[2], *[r[2] for r in refs if r]]) if rows else "(no scorecard row)"
    gap = None if None in (f["f1"], f["floor"], f["target"]) else (f["f1"] - f["floor"]) / (f["target"] - f["floor"])
    rl = f.get("book_relearn")
    rl_txt = ("; book relearning peaks " + ", ".join(f"{k} {v:.2f}" for k, v in rl.items()) + f" (limit {RELEARN_MAX})") if rl else (
        f"; {f.get('note')}" if f.get("note") else "")
    lim = None if f["floor"] is None else f["floor"] + FLOOR_MARGIN
    limits = e["limits"].format(**score_cells(name))
    return f"""---
license: {LICENSE}
license_link: https://cohere.com/c4ai-cc-by-nc-license
base_model: {AM.BASE_MODEL}
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
datasets:
- {ns}/{DATASET_NAME}
tags:
- machine-unlearning
- unlearning
- muse
- tiny-aya
- lexsi-hackathon
- research
---

# {e['title']}

> **Modified version of [`{AM.BASE_MODEL}`](https://huggingface.co/{AM.BASE_MODEL}) by Cohere Labs.** Released under
> CC-BY-NC-4.0 (inherited from the base model) together with Cohere Labs' Acceptable Use Policy: **non-commercial
> research use only.** Research checkpoint from the Lexsi Labs hackathon (Track 2b, unlearning); not for production use.

## What it is

{e['what']}

Write-up: [docs/REPORT_AYA.md]({REPORT}) (GitHub, branch `aya`). Data, result JSONs, logs and traces:
[{ns}/{DATASET_NAME}](https://huggingface.co/datasets/{ns}/{DATASET_NAME}).

## How it was produced

{e['how']}

Lineage (MUSE target-model protocol): the base model does not know the book, so we fine-tuned a *target* on the
MUSE-Books forget text plus the retain corpora and a *retain-only* twin on the retain corpora alone; the unlearning
methods start from the target, and the retain-only twin is the floor. Code: [{GITHUB}]({GITHUB}).

## Results (TEST splits)

Scorecard row, section "{sec}" of `results/SCORECARD_AYA.md` (0 = never-trained floor (book) or 0% of the target's
accuracy (algebra), 1 = the target; 95% bootstrap CIs over test items, B = 2000; `*` = degenerate on the direct
prompt; `utility_rel` = utility relative to the target):

{table}

Verbatim recall (raw continuation of 25 held-out TEST passages, ROUGE-L F1, `results/audit/`): this model
**{_fmt(f['f1'], 4)}**, retain-only floor {_fmt(f['floor'], 4)}, target (not published) {_fmt(f['target'], 4)}
(= {_fmt(gap, 3)} of the target-floor gap).
{verbatk_md(name)}{auditkit_md(name)}
## Publication check (copyright gate, `scripts/hf_publish.py`)

Published only because it is on the authors' allowlist and passes the gate: verbatim F1 at most floor + {FLOOR_MARGIN}
(here {_fmt(f['f1'], 4)} vs limit {_fmt(lim, 4)}){rl_txt}. Gate result: {'PASS' if ok else 'FAIL: ' + '; '.join(bad)}.
No file in this repo contains MUSE book text.

## Limitations

{limits}

{COMMON_LIMITS}

## Withheld models (not published, by decision of the authors)

{withheld_md()}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{ns}/{e['repo']}")
model = AutoModelForCausalLM.from_pretrained("{ns}/{e['repo']}", torch_dtype="bfloat16")
```

## License and attribution

Base model: [Tiny Aya Global](https://huggingface.co/{AM.BASE_MODEL}) by Cohere Labs, CC-BY-NC-4.0. This checkpoint is
a modified version (fine-tuned / unlearned / weight-edited as described above), distributed under the same
CC-BY-NC-4.0 license and subject to Cohere Labs' Acceptable Use Policy; non-commercial use only. The MUSE-Books
benchmark (Shi et al., 2024) was used for training and evaluation; its text is not redistributed here.
"""


def dataset_card(ns: str, keep: list[tuple], drop: list[tuple[str, str]]) -> str:
    counts: dict[str, int] = {}
    for _, d, _ in keep:
        top = "/".join(d.split("/")[:2]) if d.count("/") >= 2 else d.split("/")[0]
        counts[top] = counts.get(top, 0) + 1
    reasons: dict[str, int] = {}
    for _, why in drop:
        reasons[why] = reasons.get(why, 0) + 1
    models = "\n".join(f"- [{ns}/{e['repo']}](https://huggingface.co/{ns}/{e['repo']})" for e in ALLOWLIST.values())
    cfg = "\n".join(
        f"- config_name: math_{t}\n  data_files:\n"
        + "".join(f"  - split: {s}\n    path: math_splits/{t}/level*/{s}.jsonl\n" for s in ("train", "dev", "test")).rstrip("\n")
        for t in ("arithmetic", "algebra", "precalculus", "calculus"))
    return f"""---
license: {DATA_LICENSE}
language:
- en
- fr
- de
- es
tags:
- machine-unlearning
- unlearning
- audit
- tiny-aya
- lexsi-hackathon
pretty_name: Tiny Aya Global unlearning artifacts (Lexsi hackathon, Track 2b)
configs:
{cfg}
- config_name: word_problems
  data_files:
  - split: test
    path: attack_sets/word_problems/test.jsonl
  - split: dev
    path: attack_sets/word_problems/dev.jsonl
---

# Tiny Aya Global unlearning artifacts (Lexsi hackathon, Track 2b)

Everything we created for the study "Gone, or Just Hidden? Auditing unlearning of a book and a skill in Tiny Aya
Global" except the model weights (published separately) and anything containing MUSE book text. Write-up:
[docs/REPORT_AYA.md]({REPORT}); code: [{GITHUB}]({GITHUB}).

Published models (all derived from `CohereLabs/tiny-aya-global`, CC-BY-NC-4.0):

{models}

## Folders

| folder | what it is | how it was produced |
|---|---|---|
| `attack_sets/paraphrase_math/`, `translation_math/` (fr/de/es), `word_problems/` | held-out algebra attack sets: paraphrased, translated and word-problem versions of the algebra TEST items, with manifests and rejected items | written by a model from the algebra test inputs, kept only if a deterministic sympy verifier agrees (`attack_sets/_build/*.py`) |
| `attack_sets/_inputs/math_*.jsonl` | the algebra forget/retain test inputs the attack sets were built from | `attack_sets/_build/prep_inputs.py` |
| `math_splits/{{arithmetic,algebra,precalculus,calculus}}/level{{1,2,3}}/{{train,dev,test}}.jsonl` | generated math problems with exact answers (forget = algebra, the others = retain topics) | `src/lfm3u` math generator (sympy, seed 0) with leakage removal; `math_splits/precalculus/hendrycks/` = MATH precalculus items (Hendrycks et al., MIT) |
| `results/audit/` | TEST audit per model (direct, paraphrase, translation, jailbreak, tool, best-of-k, book verbatim/knowmem, utility) | `scripts/10_audit.py` |
| `results/relearn/`, `abl/`, `weight_attacks/`, `local_abl/`, `jspace/` | relearning (LoRA, all weights), abliteration, INT8/INT4/noise, direction ablation, J-lens readouts | `scripts/12_relearn.py`, `11_abliterate.py`, `20_weight_attacks.py`, `17_local_abl.py`, `05_jspace.py` |
| `results/dev/`, `results/unlearn/`, `results/undo/`, `results/finetune/` | dev-split sweep results and training logs (unlearning, UNDO distillation, target/retain fine-tunes) | `scripts/08_dev_eval.py`, `07_unlearn.py`, `15_undo.py`, `03_finetune.py` |
| `results/auditkit/` | independent re-evaluation with the hackathon's AuditKIT (lm-eval + native vLLM) | `scripts/22_auditkit_eval.py` |
| `results/surgery/` | AI-agent weight-surgery traces: every step file (`steps/*.py`, outputs `*.out`), scores, edit lists, state, certificates, box logs | the guarded harness `src/lfm3u/surgery/`; see `docs/SURGERY_AYA_*.md` on GitHub |
| `results/SCORECARD_AYA.md`, `AUDIT_AYA.md`, `scorecard.json` | the scorecard (95% bootstrap CIs) and the audit tables | `scripts/18_scorecard.py`, `scripts/13_audit_report.py` |
| `results/runs/*/` | Vast.ai run logs (`run.log` and per-job logs) | `scripts/vast/launch.py` |
| `results/vast/` | Vast launch records (offer, GPU, price; no secrets) | `scripts/vast/launch.py` |
| `results/verbatk/`, `results/VERBATK_AYA.md` | verbatim book recall under 9 prompt attacks: per-item ROUGE-L F1 per attack and the recovery table (no generations, no prompts) | `scripts/23_verbatim_attacks.py`, `scripts/verbatk_table.py` |
| `writeup/` | copy of the write-up (`REPORT_AYA.md`) and the compiled paper (`paper.pdf`); the originals are on GitHub | `docs/REPORT_AYA.md`, `paper/` |
| `figures/` | report figures | `scripts/21_report_figures.py` |
| `sweeps/` | every sweep / attack spec file (hyperparameters per run) | input to `scripts/vast/jobs/*.sh` |

File counts: {", ".join(f"{k} {v}" for k, v in sorted(counts.items()))}.

## What is NOT here, and why

MUSE-Books (Harry Potter) text is copyrighted and is not redistributed: no forget/retain passages, no verbmem or
knowmem questions, no book attack sets, no model continuations of book passages. Excluded by rule: book attack sets
and inputs, the verbatim-prompt paraphrases/translations, every `generations*/`, `probe*/`, `final_probe/` folder,
per-item result JSONL, weight deltas. Every remaining text file was scanned for any 12-word span of the full
MUSE-Books data (forget and retain text, verbmem and knowmem rows). A JSON file with a hit is kept only as a copy in
which every offending string (prompt or model continuation) is replaced by `{REDACTED}` and that passes a second scan;
any other file with a hit is dropped. Counts: {"; ".join(f"{v} x {k}" for k, v in reasons.items()) or "none"}.
Get MUSE from its authors (Shi et al., 2024, `muse-bench/MUSE-Books`).

Withheld models (not published): checkpoints that reproduce the book, relearn it, or are degenerate:

{withheld_md()}

## Licenses

Data we created (attack sets, generated math splits, results, logs, traces, figures): **CC-BY-4.0**. The MATH
precalculus items under `math_splits/precalculus/hendrycks/`: MIT (Hendrycks et al., 2021). Numbers in the results were
measured on models derived from `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0, Cohere Labs); the models themselves are
CC-BY-NC-4.0 (non-commercial).
"""


# ---------------------------------------------------------------------------------------------- dataset plan
DATASET_SOURCES = [  # (root, glob relative to root)
    *[(ROOT, f"attack_data/{d}/**/*") for d in ("paraphrase_math", "translation_math", "word_problems")],
    (ROOT, "attack_data/_build/*.py"), (ROOT, "attack_data/_inputs/math_*.jsonl"), (ROOT, "attack_data/_inputs/manifest_math.json"),
    *[(MAIN, f"data/{t}/**/*.jsonl") for t in ("arithmetic", "algebra", "precalculus", "calculus")],
    (ROOT, "results/audit/*/audit.json"), (ROOT, "results/audit/summary.json"),
    (ROOT, "results/relearn/*/relearn.json"), (ROOT, "results/abl/*/abl.json"),
    (ROOT, "results/weight_attacks/*/*.json"), (ROOT, "results/local_abl/*/*.json"),
    (ROOT, "results/dev/*/dev.json"), (ROOT, "results/dev/summary.csv"),
    (ROOT, "results/jspace/**/*.json"), (ROOT, "results/jspace/**/*.png"),
    (ROOT, "results/auditkit/*/*.json"), (ROOT, "results/auditkit/*/*.md"), (ROOT, "results/auditkit/*/runs/*.json"),
    (ROOT, "results/auditkit/failed.txt"),
    (ROOT, "results/unlearn/*/train.json"), (ROOT, "results/unlearn/*.json"), (ROOT, "results/undo/*/train.json"),
    (ROOT, "results/finetune/**/*.json"),
    (ROOT, "results/base/baseline.json"), (ROOT, "results/target_A15/baseline.json"), (ROOT, "results/retain_A15/baseline.json"),
    (ROOT, "results/verbatk/*/verbatk.json"), (ROOT, "results/verbatk/*.json"),
    (ROOT, "results/scorecard.json"), (ROOT, "results/aya_models.json"), (ROOT, "results/*_AYA.md"),
    (ROOT, "results/runs/*/**/*.log"), (ROOT, "results/runs/*/**/*.txt"), (ROOT, "results/vast/*.json"), (ROOT, "results/vast/*.log"),
    *[(ROOT, f"results/surgery/aya_l2_*/{g}") for g in (
        "certificate.md", "dev.json", "state.json", "edits.jsonl", "events.jsonl", "steps/**/*.py", "steps/**/*.out",
        "steps/**/*.json", "scores/**/*.json", "lib/**/*.py")],
    (ROOT, "results/surgery/aya_book_box_logs/*"),
    (ROOT, "docs/figures/*.png"), (ROOT, "scripts/sweeps/aya_*.txt"),
    (ROOT, "docs/REPORT_AYA.md"), (ROOT, "paper/main.pdf"),  # write-up copies (GitHub has the originals)
]
EXCLUDE_RE = re.compile(
    r"(^attack_data/book_questions/|^attack_data/_inputs/book_|verbatim_paraphrase|(^|/)generations[^/]*/|(^|/)probe[^/]*/|"
    r"/final_probe/|/final_math/|/deltas/|(^|/)sm_[^/]*/|/_smoke/|(^|/)_tmp/|\.pt$|\.safetensors$|"
    r"(verbmem|knowmem|excerpt)[^/]*$|/runs/book_|(^|/)book[^/]*\.jsonl$|\.env|secret|credential|token)", re.I)
JSONL_OK_RE = re.compile(r"^(attack_data/|data/|results/surgery/aya_l2_[^/]+/(edits|events)\.jsonl$)")


def pdf_text(p: Path) -> str | None:
    """Text of a PDF for the book scan; a PDF that cannot be read is treated as a hit (dropped)."""
    try:
        from pypdf import PdfReader

        return "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    except Exception:  # unreadable: refuse rather than publish unscanned
        return None


def dest_path(root: Path, rel: str) -> str:
    if root == MAIN:
        return "math_splits/" + rel.removeprefix("data/")
    for a, b in (("attack_data/", "attack_sets/"), ("docs/figures/", "figures/"), ("scripts/sweeps/", "sweeps/"),
                 ("docs/REPORT_AYA.md", "writeup/REPORT_AYA.md"), ("paper/main.pdf", "writeup/paper.pdf")):
        if rel.startswith(a):
            return b + rel[len(a):]
    return rel


REDACTED = "[removed: MUSE-Books text]"


def _redact(o, sh: set[int]):
    """Replace every string that contains a 12-word book span (prompt, continuation) by REDACTED."""
    if isinstance(o, str):
        return REDACTED if book_hit(o.encode("utf-8"), ".txt", sh) else o
    if isinstance(o, dict):
        return {k: _redact(v, sh) for k, v in o.items()}
    if isinstance(o, list):
        return [_redact(v, sh) for v in o]
    return o


def dataset_plan(sh: set[int]) -> tuple[list[tuple[Path, str, bytes | None]], list[tuple[str, str]]]:
    """keep: (source, destination, sanitised bytes or None = copy as is); drop: (path, reason). A JSON file with a book
    hit is kept only as a redacted copy (every offending string replaced) that passes a second scan."""
    keep, drop, seen = [], [], set()
    for root, pat in DATASET_SOURCES:
        for p in sorted(root.glob(pat)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            rel = p.relative_to(root).as_posix()
            tag = ("main:" if root == MAIN else "") + rel
            if EXCLUDE_RE.search(rel):
                drop.append((tag, "excluded by rule (book data, generations, probes, deltas, smoke tests)"))
            elif p.suffix == ".jsonl" and not JSONL_OK_RE.search(rel):
                drop.append((tag, "per-item result JSONL (excluded by rule)"))
            elif p.suffix.lower() == ".pdf" and ((t := pdf_text(p)) is None or not t.strip() or book_hit(t.encode("utf-8"), ".txt", sh)):
                drop.append((tag, "contains a 12-word span of the MUSE-Books data"))
            elif p.suffix.lower() not in BINARY and p.stat().st_size > MAX_SCAN_BYTES:
                drop.append((tag, "too large to scan for book text"))
            elif p.suffix.lower() not in BINARY and book_hit(p.read_bytes(), p.suffix.lower(), sh):
                data = None
                if p.suffix.lower() == ".json":
                    try:
                        data = json.dumps(_redact(json.loads(p.read_text(encoding="utf-8")), sh), indent=1,
                                          ensure_ascii=False).encode("utf-8")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        data = None
                if data is not None and not book_hit(data, ".json", sh):
                    keep.append((p, dest_path(root, rel), data))
                    drop.append((tag, "book-text strings redacted (sanitised copy kept)"))
                else:
                    drop.append((tag, "contains a 12-word span of the MUSE-Books data"))
            else:
                keep.append((p, dest_path(root, rel), None))
    return keep, drop


# ---------------------------------------------------------------------------------------------- helpers
def s3_objects(uri: str) -> list[dict]:
    from lfm3u import storage

    b, pre = storage.split_uri(uri)
    return [o for pg in storage.client().get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre.rstrip("/") + "/")
            for o in pg.get("Contents", [])]


def s3_get(bucket: str, key: str) -> bytes:
    from lfm3u import storage

    return storage.client().get_object(Bucket=bucket, Key=key)["Body"].read()


def retry(fn, what: str, n: int = 4):
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # network hiccups on 6.7 GB uploads
            if i == n - 1:
                raise
            print(f"      {what} failed ({type(e).__name__}: {str(e)[:200]}); retry {i + 1}", flush=True)
            time.sleep(20 * (i + 1))


def anon_get(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "hf-publish-verify"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def model_entries(a) -> list[dict]:
    names = [n for n in a.models.split(",") if n] if a.models else list(ALLOWLIST)
    out = []
    for n in names:
        ok, bad, facts = gate(n)
        e = {"name": n, "ok": ok, "bad": bad, "facts": facts}
        if n in ALLOWLIST:
            e |= {"repo": f"{a.namespace}/{ALLOWLIST[n]['repo']}", "s3": AM.info(n).get("s3")}
        out.append(e)
    return out


# ---------------------------------------------------------------------------------------------- steps
def step_cards(a) -> None:
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    for n in ALLOWLIST:
        (CARD_DIR / f"{n}.md").write_text(render_card(n, a.namespace), encoding="utf-8")
        print(f"   wrote {(CARD_DIR / f'{n}.md').relative_to(ROOT).as_posix()}")


def step_bundle(a) -> None:
    """Scan the small checkpoint files on S3 with the book guard, record approved sha256s, and (with --execute) put the
    cards + gate inputs + approvals on S3 for the upload box."""
    from lfm3u import storage

    sh = book_shingles()
    approved: dict[str, dict[str, str]] = {}
    for e in model_entries(a):
        if not e["ok"] or not e.get("s3"):
            print(f"   SKIP {e['name']}: {'; '.join(e['bad'])}")
            continue
        b, _ = storage.split_uri(e["s3"])
        approved[e["name"]] = {}
        for o in s3_objects(e["s3"]):
            fn = o["Key"].rsplit("/", 1)[-1]
            if not MODEL_FILE_RE.match(fn) or WEIGHT_RE.search(fn):
                continue
            data = s3_get(b, o["Key"])
            hit = book_hit(data, Path(fn).suffix, sh)
            print(f"   {e['name']}/{fn}: {len(data):,} B, {'BOOK TEXT: not approved' if hit else 'clean'}")
            if not hit:
                approved[e["name"]][fn] = hashlib.sha256(data).hexdigest()
    HF_DIR.mkdir(exist_ok=True)
    APPROVED.write_text(json.dumps(approved, indent=1), encoding="utf-8")
    missing = [n for n in approved if not (CARD_DIR / f"{n}.md").exists()]
    if missing:
        sys.exit(f"render the cards first (--what cards): missing {missing}")
    files = [APPROVED, *sorted(CARD_DIR.glob("*.md")), ROOT / "results/scorecard.json", ROOT / "results/SCORECARD_AYA.md",
             *[AM.path("audit", AM.run_name("audit", n)) for n in sorted({*approved, "retain", "target"})]]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    print(f"   bundle: {len(files)} files, {len(buf.getvalue()):,} B -> {BUNDLE_URI}")
    if a.execute:
        storage.put_bytes(buf.getvalue(), BUNDLE_URI)
        print("   uploaded")


def step_models(a, api) -> list[str]:
    approved = json.loads(APPROVED.read_text(encoding="utf-8")) if APPROVED.exists() else {}
    repos = []
    for e in model_entries(a):
        if not e["ok"]:
            print(f"   REFUSED {e['name']}: {'; '.join(e['bad'])}")
            continue
        card = CARD_DIR / f"{e['name']}.md"
        f = e["facts"]
        print(f"   {e['name']}: {e['s3']}\n      -> https://huggingface.co/{e['repo']}  (verbatim F1 {f['f1']:.4f}, floor "
              f"{f['floor']:.4f}; book relearn {f.get('book_relearn') or 'n/a'}; card {'ok' if card.exists() else 'MISSING'}; "
              f"approved small files {len(approved.get(e['name'], {}))})", flush=True)
        repos.append(e["repo"])
        if not a.execute:
            continue
        if not card.exists() or e["name"] not in approved:
            print("      not uploaded: card or approved-file list missing (run --what cards and --what bundle locally)")
            repos.pop()
            continue
        from lfm3u import storage

        b, pre = storage.split_uri(e["s3"])
        pre = pre.rstrip("/") + "/"
        tmp = Path(tempfile.mkdtemp(prefix="hfpub-", dir=a.workdir or None))
        try:
            sent = []
            for o in s3_objects(e["s3"]):
                fn = o["Key"][len(pre):]
                if not MODEL_FILE_RE.match(fn):
                    print(f"      skip {fn} (not an inference/provenance file)")
                    continue
                dst = tmp / fn
                retry(lambda: storage.client().download_file(b, o["Key"], str(dst)), f"download {fn}")
                if not WEIGHT_RE.search(fn) and hashlib.sha256(dst.read_bytes()).hexdigest() != approved[e["name"]].get(fn):
                    print(f"      skip {fn} (sha256 not approved by the local book scan)")
                    dst.unlink()
                    continue
                sent.append(fn)
            if not any(WEIGHT_RE.search(x) for x in sent):
                print("      no weights downloaded; skipped")
                repos.pop()
                continue
            shutil.copy2(card, tmp / "README.md")
            api.create_repo(e["repo"], repo_type="model", private=False, exist_ok=True)
            retry(lambda: api.upload_folder(folder_path=str(tmp), repo_id=e["repo"], repo_type="model",
                                            commit_message=f"Upload {e['name']} (Lexsi hackathon Track 2b)"), "upload")
            api.update_repo_settings(e["repo"], private=False, gated=False, repo_type="model")
            print(f"      uploaded {len(sent)} files + README.md: {', '.join(sent)}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return repos


def step_dataset(a, api) -> str:
    repo = f"{a.namespace}/{DATASET_NAME}"
    print(f"   https://huggingface.co/datasets/{repo}")
    sh = book_shingles()
    print(f"   book-text guard: {len(sh):,} distinct 12-word spans from {len(book_sources())} MUSE files")
    keep, drop = dataset_plan(sh)
    size = sum(len(data) if data is not None else p.stat().st_size for p, _, data in keep)
    groups: dict[str, int] = {}
    for _, d, _ in keep:
        k = "/".join(d.split("/")[:2])
        groups[k] = groups.get(k, 0) + 1
    print(f"   keep {len(keep)} files, {size / 1e6:.1f} MB: " + ", ".join(f"{k} {v}" for k, v in sorted(groups.items())))
    reasons: dict[str, list[str]] = {}
    for rel, why in drop:
        reasons.setdefault(why, []).append(rel)
    print(f"   drop {len(drop)} files:")
    for why, rels in reasons.items():
        print(f"     {why}: {len(rels)} (e.g. {', '.join(rels[:5])})")
    HF_DIR.mkdir(exist_ok=True)
    card = dataset_card(a.namespace, keep, drop)
    (HF_DIR / "dataset_card.md").write_text(card, encoding="utf-8")
    (HF_DIR / "dataset_dropped.txt").write_text("\n".join(f"{why}\t{rel}" for rel, why in drop) + "\n", encoding="utf-8")
    (HF_DIR / "dataset_files.txt").write_text("\n".join(d + ("  (redacted copy)" if data is not None else "")
                                                     for _, d, data in keep) + "\n", encoding="utf-8")
    print("   wrote hf/dataset_card.md, hf/dataset_files.txt, hf/dataset_dropped.txt")
    if a.execute:
        tmp = Path(tempfile.mkdtemp(prefix="hfpub-ds-"))
        try:
            for p, d, data in keep:
                (tmp / d).parent.mkdir(parents=True, exist_ok=True)
                if data is None:
                    shutil.copy2(p, tmp / d)
                else:
                    (tmp / d).write_bytes(data)
            (tmp / "README.md").write_text(card, encoding="utf-8")
            api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
            retry(lambda: api.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="dataset", delete_patterns=["*"],
                                            commit_message="Upload unlearning artifacts (no MUSE book text)"), "upload")
            api.update_repo_settings(repo, private=False, gated=False, repo_type="dataset")
            print("   uploaded")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return repo


def step_collection(a, api, repos: list[str], dataset: str) -> None:
    desc = ("Lexsi hackathon Track 2b. Report: github.com/Ungter/lexsi-unlearning/blob/aya/docs/REPORT_AYA.md. "
            "Book-memorising models withheld.")[:150]
    withheld = "Withheld: " + "; ".join(f"{w.split(' (')[0]}: {r}" for w, r in WITHHELD)
    items = [(r, "model", f"{ALLOWLIST[n]['title']}. Modified Tiny Aya Global (Cohere Labs), CC-BY-NC-4.0, non-commercial."[:500])
             for n in ALLOWLIST for r in repos if r.endswith("/" + ALLOWLIST[n]["repo"])]
    items += [(dataset, "dataset", ("Attack sets, math splits, results, logs, surgery traces; no MUSE book text. " + withheld)[:500]),
              (AM.BASE_MODEL, "model", "Reference: the base model (Cohere Labs, CC-BY-NC-4.0) every checkpoint here is derived from.")]
    print(f"   '{COLLECTION_TITLE}' ({len(desc)}-char description)")
    for it, kind, _ in items:
        print(f"     {kind}: {it}")
    if not a.execute:
        return
    col = api.create_collection(title=COLLECTION_TITLE, namespace=a.namespace, description=desc, private=False, exists_ok=True)
    have = {(i.item_id, i.item_type) for i in api.get_collection(col.slug).items}
    for it, kind, note in items:
        if (it, kind) in have:
            continue
        try:
            api.add_collection_item(col.slug, item_id=it, item_type=kind, note=note, exists_ok=True)
        except Exception as ex:  # e.g. a repo not uploaded yet, or a note too long
            print(f"     could not add {it}: {type(ex).__name__}: {str(ex)[:200]}")
    api.update_collection_metadata(col.slug, private=False, description=desc)
    print(f"   https://huggingface.co/collections/{col.slug}")


def step_verify(a) -> bool:
    ok = True
    targets = [("model", f"{a.namespace}/{e['repo']}") for e in ALLOWLIST.values()] + [("dataset", f"{a.namespace}/{DATASET_NAME}")]
    for kind, repo in targets:
        page = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"
        sp, _ = anon_get(page)
        sa, body = anon_get(f"https://huggingface.co/api/{kind}s/{repo}")
        info = json.loads(body) if sa == 200 else {}
        files = [s.get("rfilename") for s in info.get("siblings", [])]
        good = sp == 200 and sa == 200 and info.get("private") is False and info.get("gated") is False
        ok &= good
        print(f"   {'OK  ' if good else 'FAIL'} {page}: page {sp}, api {sa}, private {info.get('private')}, "
              f"gated {info.get('gated')}, {len(files)} files")
    sc, body = anon_get(f"https://huggingface.co/api/collections?owner={a.namespace}")
    cols = [c for c in (json.loads(body) if sc == 200 else []) if c.get("title") == COLLECTION_TITLE]
    for c in cols:
        sp, _ = anon_get(f"https://huggingface.co/collections/{c['slug']}")
        sa, b2 = anon_get(f"https://huggingface.co/api/collections/{c['slug']}")
        ci = json.loads(b2) if sa == 200 else {}
        good = sp == 200 and ci.get("private") is False
        ok &= good
        print(f"   {'OK  ' if good else 'FAIL'} https://huggingface.co/collections/{c['slug']}: page {sp}, private "
              f"{ci.get('private')}, {len(ci.get('items', []))} items: " + ", ".join(i.get("id", "?") for i in ci.get("items", [])))
    if not cols:
        ok = False
        print(f"   FAIL no public collection titled '{COLLECTION_TITLE}' visible anonymously")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE", NAMESPACE), help="HF user or org")
    ap.add_argument("--models", default="", help="comma list (default: the whole allowlist; anything else is refused)")
    ap.add_argument("--what", default="models,dataset,collection",
                    help="comma list of: cards, bundle, models, dataset, collection, verify")
    ap.add_argument("--workdir", default="", help="scratch dir for model downloads (box: a big disk)")
    ap.add_argument("--execute", action="store_true", help="really write (HF or the S3 bundle); without it nothing is uploaded")
    a = ap.parse_args()
    what = [w for w in a.what.split(",") if w]
    env = load_env()
    token = hf_token(env)
    mode = "EXECUTE" if a.execute else "DRY RUN (nothing is uploaded; pass --execute)"
    print(f"== hf_publish: {mode}; namespace {a.namespace}; HF token {'found' if token else 'MISSING'}; "
          f"S3 credentials {'found' if os.environ.get('AWS_ACCESS_KEY_ID') else 'missing'}")
    api = None
    if a.execute and any(w in what for w in ("models", "dataset", "collection")):
        if not token:
            sys.exit("--execute needs an HF token (HF_API in .env or HF_TOKEN in the environment)")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print(f"   huggingface_hub authenticated as {api.whoami().get('name')}")
    repos = [f"{a.namespace}/{e['repo']}" for e in ALLOWLIST.values()]
    if "cards" in what:
        print("\n== cards")
        step_cards(a)
    if "bundle" in what:
        print("\n== bundle")
        step_bundle(a)
    if "models" in what:
        print("\n== models (allowlist + copyright gate)")
        repos = step_models(a, api)
        print("   withheld:\n" + withheld_md())
    dataset = f"{a.namespace}/{DATASET_NAME}"
    if "dataset" in what:
        print("\n== dataset")
        dataset = step_dataset(a, api)
    if "collection" in what:
        print("\n== collection")
        step_collection(a, api, repos, dataset)
    if "verify" in what:
        print("\n== verify (anonymous, no token)")
        print("   ALL PUBLIC AND UN-GATED" if step_verify(a) else "   SOME CHECKS FAILED")
    if not a.execute:
        print("\nDry run only: nothing was uploaded.")


if __name__ == "__main__":
    main()
