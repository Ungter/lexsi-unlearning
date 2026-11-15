"""AuditKIT (Lexsi Labs, pip `auditkit`) evaluation of one checkpoint: utility benchmarks + the two forget domains.

  python scripts/22_auditkit_eval.py --model PATH|HF_ID --name NAME [--out results/auditkit/NAME] [--data data] [--smoke]
  python scripts/22_auditkit_eval.py --compare [--root results/auditkit] [--baseline target]

Run mode, two phases, each in its own subprocess so the GPU is free between them:
  lmeval  AuditKIT's lm-evaluation-harness engine (ak.run_lmeval, lm-eval HF backend, bf16, no chat template; these are
          loglikelihood multiple-choice tasks):
            mmlu           5-shot, --mmlu-limit docs per subject (57 subjects)     headline mmlu:acc
            arc_challenge  0-shot, --arc-limit docs                                 headline arc_challenge:acc_norm
            hellaswag      0-shot, --hellaswag-limit docs                           headline hellaswag:acc_norm
  native  AuditKIT's native spine (Adapter -> Model -> Metric) on one vLLM engine (auditkit.model.vllm_gen.VLLMModel,
          subclassed only to add the project's stop tokens arch.eos_ids), greedy:
            gsm8k              AuditKIT's built-in gsm8k scenario (first --gsm8k-limit test items), chat template, an
                               instruction to end with '#### <number>' appended, scorer numeric_match (built-in)
            forget_algebra     custom dataset: --alg-per-level seeded items per level of data/algebra/level{1,2,3}/test,
                               chat template, 2048 new tokens; custom @ak.scorer algebra_correct calling
                               lfm3u.audit.mathgrade.grade_item (gold field 'answer'), plus gibberish (MINIMIZE)
            book_verbmem_test  custom dataset: data/book/verbmem_forget_test.jsonl (25 MUSE-Books verbmem passages held
                               out from unlearning), raw completion (custom adapter, no chat template), 128 new tokens;
                               AuditKIT's rouge_l (whitespace tokens, case-sensitive) and rougeL_f1_rouge_score (the
                               project's rouge-score ROUGE-L F1, the number used in docs/AYA.md) as a custom scorer
            book_verbmem_all   the same on all 100 MUSE verbmem rows (dev 15 + test 25 + train 60)
Outputs in --out: runs/<bench>.json (AuditKIT RunResult.save: headline, stats, per-sample predictions),
report.json / report.md (AuditKIT RunResult.to_dict / report_format.Report.markdown + RunResult.summary per bench),
summary.json (headline metrics, n, timings, versions, errors). A bench that fails is recorded with its error, never faked.

Compare mode (CPU only, needs only auditkit): for every bench, auditkit.model_compare.CompareResult over all models in
--root and auditkit.comparison.RunComparison(baseline, candidate) for every other model (per-metric / per-task deltas,
retention, paired-bootstrap significance, newly wrong/correct samples via RunDiff) -> <root>/compare/report.(md|json).
On the forget benches a DROP is the intended effect; AuditKIT reports deltas without a verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LMEVAL_BENCHES = ("mmlu", "arc_challenge", "hellaswag")
NATIVE_BENCHES = ("gsm8k", "forget_algebra", "book_verbmem_test", "book_verbmem_all")
ALL_BENCHES = (*LMEVAL_BENCHES, *NATIVE_BENCHES)
# bench -> headline metrics shown in summaries / tables (first = primary)
MAIN = {"mmlu": ("mmlu:acc",), "arc_challenge": ("arc_challenge:acc_norm", "arc_challenge:acc"),
        "hellaswag": ("hellaswag:acc_norm", "hellaswag:acc"), "gsm8k": ("numeric_match",),
        "forget_algebra": ("algebra_correct", "gibberish"),
        "book_verbmem_test": ("rougeL_f1_rouge_score", "rouge_l"), "book_verbmem_all": ("rougeL_f1_rouge_score", "rouge_l")}
GSM8K_SUFFIX = "\n\nSolve the problem step by step. Give the final numeric answer on the last line in the form '#### <number>'."
SMOKE = {"mmlu_limit": 1, "arc_limit": 10, "hellaswag_limit": 10, "gsm8k_limit": 8, "alg_per_level": 3, "verb_limit": 4}


def versions() -> dict:
    out = {}
    for mod in ("auditkit", "lm_eval", "torch", "transformers", "vllm"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"not importable: {type(e).__name__}"
    return out


def save_run(r, out: Path, bench: str, name: str) -> None:
    r.run_id = f"{name}/{bench}"
    (out / "runs").mkdir(parents=True, exist_ok=True)
    r.save(str(out / "runs" / f"{bench}.json"), write_provenance=False)


# ------------------------------------------------------------------ phase: lm-eval engine

def phase_lmeval(a, out: Path) -> dict:
    import auditkit as ak
    from auditkit.runspec import RunConfig

    from lfm3u import arch

    # (shots, limit, batch): Tiny Aya's 262k vocabulary makes the logits of a 5-shot MMLU batch huge (batch 32 asked
    # for 42 GiB in one tensor), so MMLU runs at --mmlu-batch-size
    plan = {"mmlu": (5, a.mmlu_limit, a.mmlu_batch_size), "arc_challenge": (0, a.arc_limit, a.batch_size),
            "hellaswag": (0, a.hellaswag_limit, a.batch_size)}
    log = {}
    for bench in [b for b in LMEVAL_BENCHES if b in a.benches]:
        shots, limit, bs = plan[bench]
        t0 = time.time()
        try:
            opts = {"dtype": "bfloat16"}
            if a.lmeval_backend == "vllm":
                opts["model_args"] = {"gpu_memory_utilization": a.gpu_mem, "max_model_len": arch.MAX_CONTEXT}
            else:
                opts["device"] = "cuda"
            r = ak.run_lmeval(bench, model=f"{a.lmeval_backend}:{a.model}",
                              config=RunConfig(num_fewshot=shots, limit=limit or None, batch_size=bs, seed=0),
                              **opts)
            # lm-eval doc ids restart per subtask (57 MMLU subjects): make sample ids unique so RunDiff/bootstrap
            # pair the right samples across models
            for p in r.predictions:
                p.sample_id = f"{p.task}/{p.sample_id}"
            save_run(r, out, bench, a.name)
            log[bench] = {"ok": True, "seconds": round(time.time() - t0, 1), "num_fewshot": shots, "limit": limit, "batch_size": bs,
                          "engine": f"lmeval/{a.lmeval_backend}", "n": len(r.predictions)}
            print(f"[{bench}] {json.dumps({k: v for k, v in r.headline.items() if k in MAIN[bench]})}", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log[bench] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:2000]}
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return log


# ------------------------------------------------------------------ phase: native spine (vLLM)

def make_vllm(a):
    from auditkit.model.vllm_gen import VLLMModel

    from lfm3u import arch

    class AyaVLLMModel(VLLMModel):
        """AuditKIT's VLLMModel plus the stop tokens the project's own generator uses (arch.eos_ids:
        <|END_RESPONSE|>, <|END_OF_TURN_TOKEN|>, eos). Prompt rendering (chat template / raw), batching and
        sampling are AuditKIT's."""

        def _ensure_llm(self) -> None:
            fresh = self._llm is None
            super()._ensure_llm()
            if fresh:
                ids = arch.eos_ids(self._llm.get_tokenizer())
                base = self._sampling_params
                self._sampling_params = lambda **kw: base(**kw, stop_token_ids=ids)

    return AyaVLLMModel(model=a.model, name=f"vllm:{a.model}", dtype="bfloat16", max_model_len=arch.MAX_CONTEXT,
                        gpu_memory_utilization=a.gpu_mem, trust_remote_code=arch.TRUST_REMOTE_CODE)


def native_datasets(a):
    """bench -> (samples, scorers, adapter, max_tokens)."""
    import auditkit as ak
    from auditkit import Sample
    from auditkit.adapter import Adapter, _gen_params
    from auditkit.metrics.generation import RogueL
    from auditkit.model import Request
    from auditkit.types import Direction

    from lfm3u import grade
    from lfm3u.audit.mathgrade import grade_item
    from lfm3u.data.splits import read_jsonl

    class RawCompletionAdapter(Adapter):
        """Plain text continuation: the prompt goes to the model without a chat template (BOS + text), as in the
        project's verbmem_raw task."""
        method = "raw_completion"

        def adapt(self, sample, config):
            return [Request(prompt=sample.input, request_type="generate",
                            params={**_gen_params(config), "apply_chat_template": False})]

    @ak.scorer(direction=Direction.MAXIMIZE, name="algebra_correct")
    def algebra_correct(sample, output):
        return float(grade_item(sample.metadata["row"], output or "")["correct"])

    @ak.scorer(direction=Direction.MINIMIZE, name="gibberish")
    def gibberish(sample, output):
        return float(grade.is_gibberish(output or ""))

    @ak.scorer(direction=Direction.MAXIMIZE, name="rougeL_f1_rouge_score")
    def rouge_project(sample, output):
        return grade.rouge_l(output or "", str(sample.target))["rougeL_f1"]

    data = Path(a.data)
    sets = {}
    if "gsm8k" in a.benches:
        from auditkit.scenarios.gsm8k import GSM8KScenario
        rows = list(GSM8KScenario().samples())[:a.gsm8k_limit]
        sets["gsm8k"] = ([Sample(input=s.input + GSM8K_SUFFIX, target=s.target, id=f"gsm8k/{i}", task="gsm8k")
                          for i, s in enumerate(rows)], ["numeric_match"], None, 768)
    if "forget_algebra" in a.benches:
        samples = []
        for lvl in (1, 2, 3):
            pool = read_jsonl(data / f"algebra/level{lvl}/test.jsonl", allow_test=True)
            random.Random(zlib.crc32(f"auditkit|algebra|{lvl}".encode())).shuffle(pool)
            for r in pool[:a.alg_per_level] if a.alg_per_level else pool:
                samples.append(Sample(input=r["instruction"], target=str(r["answer"]), id=r["id"], task=f"algebra_L{lvl}",
                                      metadata={"row": r}))
        sets["forget_algebra"] = (samples, [algebra_correct, gibberish], None, 2048)
    verb = RawCompletionAdapter()
    for bench, files in (("book_verbmem_test", ("test",)), ("book_verbmem_all", ("dev", "test", "train"))):
        if bench not in a.benches:
            continue
        rows = [r for f in files for r in read_jsonl(data / f"book/verbmem_forget_{f}.jsonl", allow_test=True)]
        rows = rows[:a.verb_limit] if a.verb_limit else rows
        sets[bench] = ([Sample(input=r["prompt"], target=r["target"], id=r["id"], task="verbmem") for r in rows],
                       [RogueL(), rouge_project], verb, 128)
    return sets


def phase_native(a, out: Path) -> dict:
    import auditkit as ak
    from auditkit.runspec import RunConfig
    from auditkit.scenario import ListScenario

    log = {}
    try:
        sets = native_datasets(a)
        model = a.native_model if a.native_model else make_vllm(a)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {b: {"ok": False, "error": f"setup: {type(e).__name__}: {e}"[:2000]} for b in NATIVE_BENCHES if b in a.benches}
    for bench, (samples, scorers, adapter, max_tokens) in sets.items():
        t0 = time.time()
        try:
            r = ak.evaluate(ListScenario(samples, name=bench), model=model, scorers=scorers, adapter=adapter,
                            config=RunConfig(temperature=0.0, max_tokens=max_tokens), cache=False)
            save_run(r, out, bench, a.name)
            log[bench] = {"ok": True, "seconds": round(time.time() - t0, 1), "engine": "native/vllm", "n": len(samples),
                          "max_tokens": max_tokens, "failed": r.failed_count}
            if r.failed_count:  # a scorer or the model failed on some samples: keep the evidence
                log[bench]["first_errors"] = [str(e)[:500] for e in r.errors[:3]]
            print(f"[{bench}] {json.dumps(r.headline)}", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log[bench] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"[:2000]}
    if hasattr(model, "unload"):
        model.unload()
    return log


# ------------------------------------------------------------------ report (from the saved runs)

def write_report(a, out: Path, log: dict) -> dict:
    from auditkit.report import RunResult
    from auditkit.report_format import Report

    runs = {b: RunResult.load(str(out / "runs" / f"{b}.json")) for b in ALL_BENCHES
            if (out / "runs" / f"{b}.json").is_file() and log.get(b, {}).get("ok")}
    summary = {"name": a.name, "model": a.model, "smoke": a.smoke, "versions": versions(), "benches": {}, "errors": {},
               "warnings": {}}
    md = [f"# AuditKIT report: {a.name}", "", f"model `{a.model}`; " + ", ".join(f"{k} {v}" for k, v in summary["versions"].items()),
          "", "| bench | metric | value | n |", "|---|---|---|---|"]
    for b, r in runs.items():
        heads = {m: r.headline[m] for m in MAIN[b] if m in r.headline}
        n = r.stats[MAIN[b][0]].count if MAIN[b][0] in r.stats else len(r.predictions)
        summary["benches"][b] = {**heads, "n": n, "failed": r.failed_count, **{k: v for k, v in log[b].items() if k != "ok"}}
        md += [f"| {b} | {m} | {v:.4f} | {n} |" for m, v in heads.items()]
        if r.failed_count:
            summary["warnings"][b] = f"{r.failed_count} failed sample(s), left out of the means: {log[b].get('first_errors')}"
            md.append(f"| {b} | WARNING | {r.failed_count} failed samples (see report.json) | |")
    for b, entry in log.items():
        if not entry.get("ok"):
            summary["errors"][b] = entry.get("error")
            md.append(f"| {b} | FAILED | {str(entry.get('error'))[:200].replace('|', '/')} | |")
    for b, r in runs.items():
        md += ["", f"## {b}", "", Report(r).markdown(), "", "```", r.summary(), "```"]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report = {"name": a.name, "model": a.model, "versions": summary["versions"], "log": log,
              "runs": {b: {k: v for k, v in r.to_dict().items() if k != "predictions"} for b, r in runs.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ compare mode

def _sig_text(m: str, s: dict) -> str:
    if "error" in s or s.get("delta") is None:
        return f"{m}: {s}"
    lo, hi = (s.get("ci") or (float("nan"), float("nan")))[:2]
    return f"{m}: delta {s['delta']:+.4f}, {int(100 * s.get('ci_level', 0.95))}% CI [{lo:+.4f}, {hi:+.4f}], p={s['p_value']:.4f} (n={s.get('n')})"


def _short(rc) -> str:
    """RunComparison.summary() without the 57 per-subject MMLU lines."""
    return "\n".join(line for line in rc.summary().splitlines() if not line.strip().startswith("mmlu_"))


def compare(a) -> None:
    from auditkit.comparison import RunComparison
    from auditkit.model_compare import CompareResult
    from auditkit.report import RunResult

    root = Path(a.root)
    names = sorted(d.name for d in root.iterdir() if (d / "runs").is_dir() and d.name != "compare")
    if a.baseline not in names:
        sys.exit(f"baseline {a.baseline!r} has no runs under {root} (have {names})")
    order = [n for n in a.order.split(",") if n in names]
    order += [n for n in names if n not in order]
    out = root / "compare"
    out.mkdir(exist_ok=True)
    md = [f"# AuditKIT comparison (baseline: {a.baseline})", "",
          "Built from the saved AuditKIT RunResults with auditkit.model_compare.CompareResult (leaderboard) and "
          "auditkit.comparison.RunComparison (baseline -> candidate deltas, retention, paired bootstrap; sample-level "
          "changes via auditkit.diff.RunDiff). On the forget benches (forget_algebra, book_verbmem_*) a drop is the "
          "intended effect of unlearning; AuditKIT reports deltas without a verdict.", ""]
    head = "| model |" + "".join(f" {b} ({MAIN[b][0]}) |" for b in ALL_BENCHES)
    md += [head, "|---|" + "---|" * len(ALL_BENCHES)]
    runs: dict[str, dict] = {}
    for n in order:
        runs[n] = {b: RunResult.load(str(p)) for b in ALL_BENCHES if (p := root / n / "runs" / f"{b}.json").is_file()}
        cells = []
        for b in ALL_BENCHES:
            r = runs[n].get(b)
            cells.append(f" {r.headline[MAIN[b][0]]:.3f} |" if r and MAIN[b][0] in r.headline else " - |")
        md.append(f"| {n} |" + "".join(cells))
    js: dict = {"baseline": a.baseline, "models": order, "benches": {}}
    for b in ALL_BENCHES:
        have = {n: runs[n][b] for n in order if b in runs[n]}
        if not have:
            continue
        cr = CompareResult(runs=have, dataset_size=max(len(r.predictions) for r in have.values()), scorers=list(MAIN[b]))
        bj = {"leaderboard": [row for row in cr.comparison_table() if row["metric"] in MAIN[b]], "pairs": {}}
        text = "\n".join(line for line in cr.summary().splitlines() if "mmlu_" not in line)
        if b in LMEVAL_BENCHES:
            # CompareResult.summary() bootstraps the first headline metric ('mmlu:acc', ...), which lm-eval predictions
            # do not carry per sample: it finds no pairs and prints p=1.0000 without testing. Replace that block.
            text = text.split("Significance (pairwise")[0] + (
                "Significance: not shown here (AuditKIT's CompareResult.summary() looks for the per-sample headline metric, "
                "which lm-eval predictions do not carry, and prints p=1.0000 without running a test). The per-pair paired "
                "bootstrap below uses lm-eval's per-sample primary score (acc) instead.")
        md += ["", f"## {b}", "", "```", text, "```"]
        base = have.get(a.baseline)
        for n, r in have.items():
            if base is None or n == a.baseline:
                continue
            rc = RunComparison(base, r)
            sig = {}
            for m in (("acc (per-sample, lm-eval primary)",) if b in LMEVAL_BENCHES else MAIN[b]):
                try:
                    sig[m] = rc.significance(None if b in LMEVAL_BENCHES else m)
                except Exception as e:  # noqa: BLE001
                    sig[m] = {"error": f"{type(e).__name__}: {e}"}
            bj["pairs"][n] = {"metric_deltas": [{**vars(d), "grade": str(getattr(d.grade, "value", d.grade))}
                                                for d in rc.metric_deltas() if d.metric in MAIN[b]],
                              "retention": {m: v for m, v in rc.retention().items() if m in MAIN[b]},
                              "significance": sig, "samples": rc.sample_summary()}
            md += ["", f"### {a.baseline} -> {n}", "", "```", _short(rc), "```", "",
                   "paired bootstrap (candidate - baseline): " + "; ".join(_sig_text(m, s) for m, s in sig.items())]
        js["benches"][b] = bj
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    print("\n".join(md[:len(order) + 6]))
    print(f"wrote {out / 'report.md'}")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--out")
    ap.add_argument("--data", default="data")
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    ap.add_argument("--phases", default="lmeval,native")
    ap.add_argument("--phase", default=None, help="internal: run one phase in this process")
    ap.add_argument("--lmeval-backend", default="hf", choices=("hf", "vllm"))
    ap.add_argument("--batch-size", default="32", help="lm-eval batch size for arc_challenge / hellaswag")
    ap.add_argument("--mmlu-batch-size", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--mmlu-limit", type=int, default=20, help="docs per MMLU subject (57 subjects)")
    ap.add_argument("--arc-limit", type=int, default=500)
    ap.add_argument("--hellaswag-limit", type=int, default=500)
    ap.add_argument("--gsm8k-limit", type=int, default=250)
    ap.add_argument("--alg-per-level", type=int, default=100, help="0 = all ~1,400 algebra test items")
    ap.add_argument("--verb-limit", type=int, default=0)
    ap.add_argument("--native-model", default=None, help="testing: an AuditKIT model spec (e.g. echo) instead of vLLM")
    ap.add_argument("--smoke", action="store_true", help=f"tiny limits {SMOKE}")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--root", default="results/auditkit")
    ap.add_argument("--baseline", default="target")
    ap.add_argument("--order", default="base,target,retain")
    a = ap.parse_args()
    if a.compare:
        return compare(a)
    if not (a.model and a.name):
        ap.error("--model and --name are required")
    if a.smoke:
        for k, v in SMOKE.items():
            setattr(a, k, v)
    a.benches = [b for b in a.benches.split(",") if b]
    a.batch_size = int(a.batch_size) if a.batch_size.isdigit() else a.batch_size
    out = Path(a.out or f"results/auditkit/{a.name}")
    out.mkdir(parents=True, exist_ok=True)

    if a.phase:  # child
        log = phase_lmeval(a, out) if a.phase == "lmeval" else phase_native(a, out)
        (out / f"phase_{a.phase}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return

    log: dict = {}
    for ph in [p for p in a.phases.split(",") if p]:
        benches = [b for b in (LMEVAL_BENCHES if ph == "lmeval" else NATIVE_BENCHES) if b in a.benches]
        if not benches:
            continue
        pf = out / f"phase_{ph}.json"
        pf.unlink(missing_ok=True)
        t0 = time.time()
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
        rc = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--phase", ph], env=env).returncode
        if pf.is_file():
            log |= json.loads(pf.read_text(encoding="utf-8"))
        for b in benches:
            log.setdefault(b, {"ok": False, "error": f"phase {ph} exited rc={rc} after {time.time() - t0:.0f}s before "
                                                     f"recording {b} (see run log)"})
    s = write_report(a, out, log)
    print(json.dumps(s, indent=2, default=str))
    sys.exit(0 if not s["errors"] else 3)


if __name__ == "__main__":
    main()
