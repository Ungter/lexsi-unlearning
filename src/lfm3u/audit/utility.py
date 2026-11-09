"""Utility evals (PLAN 1.4 / 6.4): MMLU fixed 1,000-question subset, IFEval, perplexity on held-out FineWeb.

  mmlu         cais/mmlu test, 1,000 questions drawn with a fixed seed; standard 5-shot raw-completion prompt (the
               subject's 5 dev examples, as in lm-eval), answer = argmax over the letter tokens after "Answer:" (vLLM
               allowed_token_ids), accuracy. A zero-shot chat-template version was dropped: its first-token letter choice
               was dominated by position bias (base model answered B 48% and D 4% of the time).
  ifeval       google/IFEval (541 prompts), graded with lm-eval's IFEval checkers on the final answer
               (reasoning stripped); prompt/instruction-level strict and loose accuracy
  ppl_fineweb  FineWeb docs 0-499 of s3://lexsi-hackathon/eval/fineweb_sample.jsonl (docs 500-1499 are the J-lens
               fitting corpus), first 1,024 tokens each, token-weighted perplexity
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path

MMLU = ("cais/mmlu", "all/test-00000-of-00001.parquet")
MMLU_DEV = ("cais/mmlu", "all/dev-00000-of-00001.parquet")
IFEVAL = ("google/IFEval", "ifeval_input_data.jsonl")
FINEWEB_URI = "s3://lexsi-hackathon/eval/fineweb_sample.jsonl"
PPL_DOCS = (0, 500)
LETTERS = "ABCD"


def mmlu_subset(n: int = 1000, seed: int = 0) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = pq.read_table(hf_hub_download(MMLU[0], MMLU[1], repo_type="dataset")).to_pylist()
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    return [{"idx": i, **rows[i]} for i in idx]


def mmlu_shots() -> dict[str, list[dict]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shots: dict[str, list[dict]] = {}
    for r in pq.read_table(hf_hub_download(MMLU_DEV[0], MMLU_DEV[1], repo_type="dataset")).to_pylist():
        shots.setdefault(r["subject"], []).append(r)
    return shots


def _mmlu_block(r: dict, with_answer: bool) -> str:
    opts = "\n".join(f"{L}. {str(c).strip()}" for L, c in zip(LETTERS, r["choices"]))
    return f"{r['question'].strip()}\n{opts}\nAnswer:" + (f" {LETTERS[r['answer']]}\n\n" if with_answer else "")


def mmlu_prompt(r: dict, shots: dict[str, list[dict]], k: int = 5) -> str:
    subject = r["subject"].replace("_", " ")
    head = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    return head + "".join(_mmlu_block(s, True) for s in shots.get(r["subject"], [])[:k]) + _mmlu_block(r, False)


def letter_tokens(tok) -> dict[int, str]:
    """Single-token encodings of A-D, with and without a leading space."""
    out = {}
    for L in LETTERS:
        for s in (L, " " + L):
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out[ids[0]] = L
    if set(out.values()) != set(LETTERS):
        raise RuntimeError("could not find single-token encodings for A-D")
    return out


def ifeval_items() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(IFEVAL[0], IFEVAL[1], repo_type="dataset")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _ifeval_utils():
    """lm_eval.tasks.ifeval.utils without importing lm_eval.tasks (its TaskManager pulls in the whole harness)."""
    spec = importlib.util.find_spec("lm_eval")
    if spec is None:
        raise ImportError("pip install lm-eval langdetect immutabledict nltk")
    if "lm_eval.tasks" not in sys.modules:
        pkg = types.ModuleType("lm_eval.tasks")
        pkg.__path__ = [str(Path(spec.origin).parent / "tasks")]
        sys.modules["lm_eval.tasks"] = pkg
    return importlib.import_module("lm_eval.tasks.ifeval.utils")


def grade_ifeval(doc: dict, response: str) -> dict:
    res = _ifeval_utils().process_results(doc, [response])
    return {"prompt_strict": bool(res["prompt_level_strict_acc"]), "inst_strict": [bool(x) for x in res["inst_level_strict_acc"]],
            "prompt_loose": bool(res["prompt_level_loose_acc"]), "inst_loose": [bool(x) for x in res["inst_level_loose_acc"]]}


def summarize_ifeval(rows: list[dict]) -> dict:
    inst_s = [x for r in rows for x in r["inst_strict"]]
    inst_l = [x for r in rows for x in r["inst_loose"]]
    m = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"n": len(rows), "prompt_strict": m([r["prompt_strict"] for r in rows]), "inst_strict": m(inst_s),
            "prompt_loose": m([r["prompt_loose"] for r in rows]), "inst_loose": m(inst_l)}


def fineweb_docs(cache: str | Path = "work/fineweb_sample.jsonl", span: tuple[int, int] = PPL_DOCS) -> list[dict]:
    cache = Path(cache)
    if not cache.is_file():
        from lfm3u import storage

        cache.parent.mkdir(parents=True, exist_ok=True)
        b, k = storage.split_uri(FINEWEB_URI)
        storage.client().download_file(b, k, str(cache))
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[span[0]:span[1]]


def perplexity(logprobs: list[list[float]]) -> dict:
    n = sum(len(x) for x in logprobs)
    nll = -sum(sum(x) for x in logprobs) / max(n, 1)
    per_doc = [-sum(x) / len(x) for x in logprobs if x]
    return {"n_docs": len(logprobs), "n_tokens": n, "nll": round(nll, 4), "ppl": round(math.exp(nll), 3),
            "doc_mean_ppl": round(sum(math.exp(v) for v in per_doc) / len(per_doc), 3)}
