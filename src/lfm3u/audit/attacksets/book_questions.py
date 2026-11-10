"""Verifier for the MUSE-Books attack question sets (paraphrase / translation /
indirect / excerpt questions).

``verify(rows, sources) -> (kept, rejected)``

``sources`` maps a parent id to a dict:
  * QA parents (MUSE knowmem rows): ``{"question": ..., "answer": ...}``
  * excerpt / chunk parents:        ``{"text": ...}``  (excerpt = prompt + target)

A row is kept only if all of these hold:
  (a) question non-empty and does not contain the normalised gold answer
      (lowercase, punctuation stripped, articles removed; token-sequence match);
  (b) paraphrase / indirect / translation: ``answer`` equals the parent's answer exactly;
  (c) paraphrase: token Jaccard with the original question < 0.8;
  (d) excerpt questions: ``evidence`` (whitespace-normalised) is a substring of the
      whitespace-normalised source text, has <= 30 words and contains the normalised
      answer; the answer has 1-6 words;
  (e) translations: at least two distinct function words of the target language.

CLI: ``python -m lfm3u.audit.attacksets.book_questions [--out DIR] [--inputs DIR]``
re-checks the output files against the inputs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ARTICLES = {"a", "an", "the"}
FUNCTION_WORDS = {
    "translation_fr": {"la", "le", "de", "des", "qui", "est"},
    "translation_de": {"der", "die", "das", "wer", "ist", "sie"},
    "translation_es": {"el", "la", "de", "qué", "quién", "es"},
}
QA_ATTACKS = {"paraphrase", "indirect", "translation_fr", "translation_de", "translation_es"}
EXCERPT_ATTACKS = {"excerpt_heldout", "excerpt_forget_chunk", "excerpt_retain"}
MAX_EVIDENCE_WORDS = 30
JACCARD_MAX = 0.8
REQUIRED_FIELDS = ("id", "parent_id", "split", "target", "attack", "question", "answer", "generator")


def tokens(text: str) -> list[str]:
    """Lowercase, punctuation stripped (to spaces), English articles removed."""
    toks = re.findall(r"\w+", (text or "").lower())
    return [t for t in toks if t not in ARTICLES]


def norm(text: str) -> str:
    return " ".join(tokens(text))


def contains_seq(hay: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def ws(text: str) -> str:
    return " ".join((text or "").split())


def _check(row: dict, sources: dict) -> str | None:
    """Return a rejection reason, or None if the row passes."""
    for f in REQUIRED_FIELDS:
        if f not in row or row[f] in (None, ""):
            return f"missing field {f}"
    attack = row["attack"]
    question, answer = row["question"], row["answer"]
    if not str(question).strip():
        return "empty question"
    qt = tokens(question)
    at = tokens(answer)
    if not at:
        return "empty normalised answer"
    # (a) answer must not leak into the question
    if contains_seq(qt, at):
        return "question contains answer"
    if attack in QA_ATTACKS and row.get("answer_translated"):
        if contains_seq(qt, tokens(row["answer_translated"])):
            return "question contains translated answer"
    parent = sources.get(row["parent_id"])
    if parent is None:
        return "unknown parent_id"
    if attack in QA_ATTACKS:
        # (b) same gold answer as the parent
        if answer != parent.get("answer"):
            return "answer differs from parent"
        orig = row.get("question_original")
        if not orig or orig != parent.get("question"):
            return "question_original differs from parent question"
        if attack == "paraphrase":
            j = jaccard(question, orig)
            if j >= JACCARD_MAX:
                return f"paraphrase too close (jaccard {j:.2f})"
        if attack == "indirect" and ws(question) == ws(orig):
            return "indirect identical to original"
        if attack.startswith("translation_"):
            if ws(question) == ws(orig):
                return "translation identical to English"
            if not row.get("answer_translated"):
                return "missing answer_translated"
            words = set(re.findall(r"\w+", question.lower()))
            hits = words & FUNCTION_WORDS[attack]
            if len(hits) < 2:
                return f"language check failed ({len(hits)} function words)"
    elif attack in EXCERPT_ATTACKS:
        # (d) evidence grounded in the source text
        ev = ws(row.get("evidence", ""))
        if not ev:
            return "missing evidence"
        if len(ev.split()) > MAX_EVIDENCE_WORDS:
            return "evidence longer than 30 words"
        text = ws(parent.get("text", ""))
        if ev not in text:
            return "evidence not a substring of source"
        if not contains_seq(tokens(ev), at):
            return "evidence does not contain answer"
        if not 1 <= len(str(answer).split()) <= 6:
            return "answer not 1-6 words"
    else:
        return f"unknown attack {attack}"
    return None


def verify(rows: list[dict], sources: dict) -> tuple[list[dict], list[dict]]:
    kept, rejected = [], []
    seen: set[str] = set()
    for row in rows:
        reason = _check(row, sources)
        if reason is None and row["id"] in seen:
            reason = "duplicate id"
        if reason is None:
            seen.add(row["id"])
            kept.append(row)
        else:
            rejected.append({**row, "reject_reason": reason})
    return kept, rejected


# ----------------------------------------------------------------------------
# IO helpers / CLI
# ----------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_sources(inputs: Path) -> dict:
    src: dict[str, dict] = {}
    for name in ("book_knowmem_forget_test", "book_knowmem_forget_dev", "book_knowmem_retain_test"):
        for r in read_jsonl(inputs / f"{name}.jsonl"):
            src[r["id"]] = {"question": r["question"], "answer": r["answer"]}
    for r in read_jsonl(inputs / "book_passages_heldout.jsonl"):
        src[r["id"]] = {"text": r["prompt"] + r["target"]}
    for name in ("book_chunks_forget", "book_chunks_retain_test"):
        for r in read_jsonl(inputs / f"{name}.jsonl"):
            src[r["id"]] = {"text": r["text"]}
    return src


OUTPUT_FILES = ("paraphrase", "translation", "indirect", "excerpt_questions")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="attack_data/book_questions")
    ap.add_argument("--inputs", default="attack_data/_inputs")
    args = ap.parse_args(argv)
    out, inputs = Path(args.out), Path(args.inputs)
    sources = load_sources(inputs)
    total_bad = 0
    for name in OUTPUT_FILES:
        path = out / f"{name}.jsonl"
        if not path.exists():
            print(f"{name}: MISSING {path}")
            total_bad += 1
            continue
        rows = read_jsonl(path)
        kept, rejected = verify(rows, sources)
        by_attack: dict[str, int] = {}
        for r in kept:
            key = f"{r['attack']}/{r['split']}"
            by_attack[key] = by_attack.get(key, 0) + 1
        print(f"{name}: rows={len(rows)} kept={len(kept)} rejected={len(rejected)} {json.dumps(by_attack, sort_keys=True)}")
        for r in rejected:
            print(f"  REJECT {r['id']}: {r['reject_reason']}")
        total_bad += len(rejected)
    rej_path = out / "rejected.jsonl"
    n_rej = len(read_jsonl(rej_path)) if rej_path.exists() else 0
    print(f"rejected.jsonl rows (build-time rejects, kept out of the sets): {n_rej}")
    print("VERIFIER:", "PASS" if total_bad == 0 else f"FAIL ({total_bad} problems)")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
