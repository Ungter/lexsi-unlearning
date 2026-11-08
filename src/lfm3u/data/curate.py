import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path

import sympy as sp

from .math_gen import SUFFIX, parse_expression, same_expr
from .splits import (
    ensure_writable,
    read_jsonl,
    sha256,
    source_id,
    split_counts,
    write_json,
    write_jsonl,
)

LANGUAGES = ("French", "German", "Spanish")
ALPACA_ID = "tatsu-lab/alpaca"
PARAPHRASE_TEMPLATE = (
    "Reword the question. Keep all mathematics, facts, and the requested operation identical.\n"
    "Ignore the evolution strategy if it would change the task. Strategy: {strategy}\n"
    "Context: {context}\nQuestion: {instruction}\n"
    'Return JSON only: {{"evolved_instruction": "<reworded question>"}}'
)
WORD_PROBLEM_TEMPLATE = (
    "Rewrite this task as a word or physics problem with a hidden calculus step. "
    "Preserve the requested operation, variable, expression, and bounds. Do not give the answer.\n"
    "Strategy: {strategy}\nContext: {context}\nTask: {instruction}\n"
    "The rewritten instruction must have exactly two lines:\n"
    "Problem: <the problem>\nExpression: <the original underlying expression in SymPy syntax>\n"
    'Return JSON only: {{"evolved_instruction": "<the two lines, separated by an escaped newline>"}}'
)
TRANSLATE_TEMPLATE = (
    "Translate the question into {language}. Keep every symbol, number, name, and task identical.\n"
    "Strategy (do not change difficulty): {strategy}\nContext: {context}\nQuestion: {instruction}\n"
    'Return JSON only: {{"evolved_instruction": "<translated question>"}}'
)
QUESTION_REWARD_TEMPLATE = (
    "Score only whether this rewritten question is coherent, precise, self-contained, and answerable. "
    "Do not penalize the absence of an answer. Question: {instruction}\nUnused answer: {response}\n"
    'Return JSON: {{"overall_score": 0.0, "reasoning": "<short reason>"}} with score from 0 to 1.'
)
EXPR_RE = re.compile(r"^\s*Expression:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _models():
    llm = os.environ.get("CURATOR_LLM_MODEL")
    judge = os.environ.get("CURATOR_JUDGE_MODEL")
    if not llm or not judge:
        raise RuntimeError("set CURATOR_LLM_MODEL and CURATOR_JUDGE_MODEL in the environment or .env")
    return llm, judge


def _record(value):
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")


def _normal(text):
    return " ".join(str(text).split())


def _source_index(sources):
    index = {"ids": {}, "instructions": {}, "contexts": {}}
    for source in sources:
        for key in {source.get("id"), source.get("source_id")} - {None, ""}:
            index["ids"].setdefault(str(key), []).append(source)
        instruction = _normal(source.get("instruction", ""))
        if instruction:
            index["instructions"].setdefault(instruction, []).append(source)
        context = _normal(source.get("text") or source.get("input", ""))
        if context:
            index["contexts"].setdefault(context, []).append(source)
    return index


def _resolve(row, index):
    metadata = row.get("metadata") or {}
    reference = metadata.get("source_sample_id") or row.get("parent_id") or row.get("id")
    candidates = index["ids"].get(str(reference), [])
    if not candidates and metadata.get("original_instruction"):
        candidates = index["instructions"].get(_normal(metadata["original_instruction"]), [])
    if not candidates:
        context = row.get("input") or row.get("text")
        if context:
            candidates = index["contexts"].get(_normal(context), [])
    if not candidates:
        candidates = index["instructions"].get(_normal(row.get("instruction", "")), [])
    if len(candidates) != 1:
        raise ValueError("generated sample has missing or ambiguous source provenance")
    return candidates[0]


def _curator(dataset, output_dir, task, template=None, *, seed=0, **extra):
    from curatorkit import Curator, CuratorConfig

    output_dir = ensure_writable(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"curation output is not empty: {output_dir}")
    sources = read_jsonl(dataset)
    index = _source_index(sources)
    cfg = {"dataset": str(dataset), "generation_task": task, "dedup": "minhash", "clean": False,
           "min_tokens": 1, "export_formats": ["alpaca"], "output_dir": str(output_dir), "minhash_seed": seed}
    if task is not None:
        llm, judge = _models()
        cfg.update(llm_model=llm, judge_llm_model=judge, generate_answers=False, reward_threshold=0.7,
                   judge_on_error="reject")
    if task == "evol":
        cfg.update(num_evolutions=1, evol_prompt_template=template, schema_gate=False,
                   reward_prompt_template=QUESTION_REWARD_TEMPLATE)
    cfg.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    if task == "adversarial_qa":
        connector_rows = [{"id": row.get("id", source_id(row)), "text": row["text"]} for row in sources]
    else:
        connector_rows = [{"id": row.get("id", source_id(row)), "instruction": row.get("instruction", ""),
                           "input": row.get("input", ""), "output": row.get("output", "")} for row in sources]
    connector_source = write_jsonl(output_dir / "connector_source.jsonl", connector_rows)
    cfg["dataset"] = str(connector_source)
    manifest = {"status": "running", "source": str(Path(dataset).resolve()), "source_sha256": sha256(dataset),
                "task": task, "seed": seed, "models": {k: cfg[k] for k in ("llm_model", "judge_llm_model") if k in cfg},
                "template_sha256": hashlib.sha256((template or "").encode("utf-8")).hexdigest(),
                "dedup": "minhash", "reward_threshold": cfg.get("reward_threshold")}
    write_json(output_dir / "manifest.json", manifest)
    generation_cfg = {**cfg, "reward_threshold": None} if task == "evol" else cfg
    result = Curator(CuratorConfig(**generation_cfg)).run()
    rejected = [_record(row) for row in result.rejected]
    generated_records = {}
    if task == "evol" and result.passed:
        judging_rows = []
        generated_sources = {}
        for sample in result.passed:
            row = _record(sample)
            try:
                source = _resolve(row, index)
            except ValueError as exc:
                rejected.append({"sample": row, "reason": str(exc)})
                continue
            row["output"] = source.get("output") or source.get("answer", "")
            if not row["output"]:
                rejected.append({"sample": row, "reason": "source answer is empty"})
                continue
            generated_sources[row["id"]] = source
            generated_records[row["id"]] = row
            judging_rows.append({"id": row["id"], "instruction": row["instruction"],
                                 "input": "", "output": row["output"]})
        judging_path = write_jsonl(output_dir / "judging_source.jsonl", judging_rows)
        quality = Curator(CuratorConfig(
            dataset=str(judging_path), format="alpaca", generation_task=None, clean=False,
            dedup="none", schema_gate=False, llm_model=cfg["llm_model"], judge_llm_model=cfg["judge_llm_model"],
            reward_threshold=0.7, reward_prompt_template=QUESTION_REWARD_TEMPLATE,
            judge_on_error="reject", export_formats=["alpaca"], output_dir=str(output_dir / "quality"),
        )).run()
        rejected.extend(_record(row) for row in quality.rejected)
        result = quality
        for sample in result.passed:
            row = _record(sample)
            if row.get("id") in generated_sources:
                index["ids"][row["id"]] = [generated_sources[row["id"]]]
    accepted, provenance = [], []
    for sample in result.passed:
        row = _record(sample)
        if row.get("id") in generated_records:
            original = generated_records[row["id"]]
            row["metadata"] = original.get("metadata", {})
            row["provenance_chain"] = original.get("provenance_chain", []) + row.get("provenance_chain", [])
        try:
            source = _resolve(row, index)
        except ValueError as exc:
            rejected.append({"sample": row, "reason": str(exc)})
            continue
        instruction = str(row.get("instruction") or "").strip()
        answer = source.get("output", "") if task == "evol" else row.get("output", "")
        if not instruction or (task != "evol" and not str(answer).strip()):
            rejected.append({"sample": row, "reason": "empty instruction or answer"})
            continue
        sid = source_id(source)
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]
        accepted.append({**source, "instruction": instruction, "question": instruction, "input": "",
                         "output": answer, "answer": str(answer).removeprefix("Answer: ").strip(),
                         "id": f"{sid}-{task or 'clean'}-{digest}", "source_id": sid,
                         "parent_id": source.get("id", sid), "variant": task or "clean"})
        provenance.append({"id": accepted[-1]["id"], "source_id": sid, "parent_id": source.get("id", sid),
                           "split": source.get("split"), "generated_id": row.get("id"),
                           "chain": row.get("provenance_chain", []), "metadata": row.get("metadata", {})})
    write_jsonl(output_dir / "accepted.jsonl", accepted)
    write_jsonl(output_dir / "provenance.jsonl", provenance)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    manifest["curatorkit"] = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(status="complete", accepted=len(accepted), rejected=len(rejected),
                    files={name: sha256(output_dir / name)
                           for name in ("accepted.jsonl", "provenance.jsonl", "rejected.jsonl")})
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def read_exports(output_dir):
    output_dir = Path(output_dir)
    for name in ("accepted.jsonl", "sft_alpaca.jsonl"):
        path = output_dir / name
        if path.is_file():
            return read_jsonl(path)
    raise FileNotFoundError(f"no accepted Alpaca export in {output_dir}")


def write_curation_source(items, path):
    rows = []
    for item in items:
        instruction = item.get("instruction") or item.get("question") or item.get("prompt") or ""
        output = item.get("output") or item.get("answer") or item.get("target") or ""
        rows.append({**item, "instruction": instruction, "output": output,
                     "id": item.get("id", source_id(item)), "source_id": source_id(item)})
    return write_jsonl(path, rows)


def paraphrase(source, output_dir, *, seed=0):
    return _curator(source, output_dir, "evol", PARAPHRASE_TEMPLATE, seed=seed)


def verify_word_problems(rows):
    kept, dropped = [], []
    for row in rows:
        text = row.get("instruction", "")
        match = EXPR_RE.search(text)
        expected = row.get("expr") or row.get("context")
        if match and expected and same_expr(match.group(1).strip().strip("`"), expected):
            problem = EXPR_RE.sub("", text).strip()
            problem = re.sub(r"^Problem:\s*", "", problem, flags=re.IGNORECASE).strip()
            if problem:
                kept.append({**row, "instruction": problem.removesuffix(SUFFIX) + SUFFIX,
                             "question": problem, "underlying_expression": match.group(1).strip()})
                continue
        dropped.append({**row, "rejection_reason": "missing problem or non-equivalent underlying expression"})
    return kept, dropped


def word_problems(source, output_dir, *, seed=0):
    out = _curator(source, output_dir, "evol", WORD_PROBLEM_TEMPLATE, seed=seed)
    kept, dropped = verify_word_problems(read_exports(out))
    write_jsonl(out / "verified.jsonl", kept)
    write_jsonl(out / "sympy_rejected.jsonl", dropped)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(verified=len(kept), sympy_rejected=len(dropped))
    manifest["files"].update({name: sha256(out / name) for name in ("verified.jsonl", "sympy_rejected.jsonl")})
    write_json(out / "manifest.json", manifest)
    return {"kept": len(kept), "dropped": len(dropped)}


def notation_variants(item):
    kind, expr = item.get("kind"), item.get("expr")
    if not expr:
        return []
    tex = sp.latex(parse_expression(expr))
    if kind == "diff":
        return [f"Find d/dx [{expr}].", f"Let f(x) = {expr}. Compute f′(x).",
                f"If y = {expr}, find dy/dx.", f"What is the rate of change of {expr} with respect to x?",
                f"Differentiate ${tex}$ with respect to x."]
    if kind == "integral":
        return [f"Evaluate ∫ ({expr}) dx.", f"Find ∫ {expr} dx.",
                f"Compute the indefinite integral of {expr} with respect to x.",
                f"Find a function whose derivative with respect to x is ${tex}$."]
    if kind == "defint":
        lower, upper = item.get("lower"), item.get("upper")
        if lower == "" or upper == "" or lower is None or upper is None:
            return []
        return [f"Evaluate ∫ from {lower} to {upper} of ({expr}) dx.",
                f"Compute $\\int_{{{lower}}}^{{{upper}}} {tex}\\,dx$."]
    return []


def notation_rows(items):
    rows = []
    for item in items:
        for variant in notation_variants(item):
            digest = hashlib.sha256(variant.encode("utf-8")).hexdigest()[:16]
            rows.append({**item, "instruction": variant + SUFFIX, "question": variant,
                         "output": f"Answer: {item['answer']}", "source_id": source_id(item),
                         "parent_id": item["id"], "id": f"{item['id']}-notation-{digest}", "variant": "notation"})
    return rows


def adversarial_book(raw_path, output_dir, *, seed=0):
    return _curator(raw_path, output_dir, "adversarial_qa", generate_answers=True,
                    difficulty="hard", hallucination_threshold=0.7, format="pretrain", seed=seed)


def translations(source, root, languages=LANGUAGES, *, seed=0):
    rows = read_jsonl(source)
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("translations are restricted to explicit test-split items")
    if any(lang not in LANGUAGES for lang in languages):
        raise ValueError(f"supported translation languages are {LANGUAGES}")
    return {lang: _curator(source, Path(root) / lang.lower(), "evol",
                          TRANSLATE_TEMPLATE.replace("{language}", lang), seed=seed) for lang in languages}


def general_retain(n=2000, seed=0, *, output_dir=None):
    from datasets import load_dataset

    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="lfm3u-retain-") as directory:
            return general_retain(n, seed, output_dir=directory)
    output_dir = ensure_writable(output_dir)
    rows = []
    for row in load_dataset(ALPACA_ID, split="train"):
        instruction, output = str(row.get("instruction") or "").strip(), str(row.get("output") or "").strip()
        context = str(row.get("input") or "").strip()
        if not instruction or not output:
            continue
        instruction += "\n" + context if context else ""
        sid = "alpaca-" + hashlib.sha256(_normal(instruction).encode("utf-8")).hexdigest()[:20]
        rows.append({"id": sid, "source_id": sid, "instruction": instruction, "output": output,
                     "dataset": ALPACA_ID})
    unique = {row["source_id"]: row for row in rows}
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    candidate_count = min(len(rows), max(2 * n, n + 32))
    round_number = 0
    while True:
        source = write_jsonl(Path(output_dir) / f"source_{round_number}.jsonl", rows[:candidate_count])
        out = _curator(source, Path(output_dir) / f"curation_{round_number}", None,
                       seed=seed, clean=True)
        kept = read_exports(out)
        if len(kept) >= n:
            return kept[:n]
        if candidate_count == len(rows):
            raise ValueError(f"only {len(kept)} retain items survived cleaning; requested {n}")
        candidate_count = min(len(rows), max(candidate_count + 1, 2 * candidate_count))
        round_number += 1


def build_general_retain(root, n=2000, seed=0):
    root = ensure_writable(root)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("the retain training count must be positive")
    sizes = {"train": n, "dev": (n + 3) // 4, "test": (5 * n + 11) // 12}
    rows = general_retain(sum(sizes.values()), seed, output_dir=Path(root) / "curation" / "general_retain")
    counts = {}
    for name, part in split_counts(rows, sizes, seed=seed).items():
        write_jsonl(Path(root) / "general" / f"retain_{name}.jsonl", (dict(row, split=name) for row in part))
        counts[f"general/retain_{name}"] = len(part)
    return counts
