"""Verifier for the LLM-written math paraphrase attack set (attack_data/paraphrase_math/).

verify(rows) -> (kept, rejected). Rejected rows get a `reject_reasons` list. Rows are output rows: all parent fields
plus question (paraphrase), question_original, instruction.
"""
from __future__ import annotations

import json
import re
import sys

SUFFIX = " Give the final answer on the last line as 'Answer: <expression>'."
JACCARD_MAX = 0.8

_KEYWORDS = {
    "diff": ["derivative", "differentiate", "d/dx", "rate of change", "f'"],
    "integral": ["integral", "antiderivative", "integrate", "primitive"],
    "defint": ["integral", "integrate"],
    "solve": ["solve", "solution", "find", "determine", "roots", "value of x", "satisfy", "satisfies"],
    "quadratic": ["solve", "solution", "find", "determine", "roots", "zeros", "zeroes", "satisfy"],
    "factor": ["factor", "factorise", "factorize", "factorisation", "factorization", "product of"],
    "algebra": ["expand", "multiply out", "multiply", "product", "distribute", "multiplying", "expansion"],
    "simplify": ["simplify", "simplified", "reduce", "simplest", "cancel", "simpler"],
    "trig": ["compute", "evaluate", "calculate", "value", "find", "determine", "what is", "work out"],
    "log": ["compute", "evaluate", "calculate", "value", "find", "determine", "what is", "work out"],
    "identity": ["simplify", "simplified", "reduce", "simplest", "simpler", "equivalent", "evaluate", "express"],
}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def _numbers(s: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", s)


def _has_token(text: str, tok: str, num: bool = False) -> bool:
    # num=True: only digits/decimal points delimit (so the 2 in log_2 counts); else word chars delimit too
    pre = r"(?<![\d.])" if num else r"(?<![\w.])"
    post = r"(?!\d|\.\d)" if num else r"(?!\w|\.\d)"
    return re.search(pre + re.escape(tok) + post, text) is not None


def check_row(r: dict) -> list[str]:
    bad = []
    ins = r.get("instruction", "")
    kind = r.get("kind", "")
    expr = r.get("expr", "") or ""
    orig = r.get("question_original", "")
    para = r.get("question", "")
    if expr:
        if expr not in ins:
            bad.append("expr_not_verbatim")
    else:
        miss = [n for n in set(_numbers(orig)) if not _has_token(ins, n, True)]
        if miss:
            bad.append(f"numbers_missing:{sorted(miss)}")
    if kind == "defint":
        rest = ins.replace(expr, " ") if expr else ins
        for k in ("lower", "upper"):
            if not _has_token(rest, str(r.get(k, ""))):
                bad.append(f"{k}_bound_missing")
    if not ins.endswith(SUFFIX):
        bad.append("bad_suffix")
    elif ins[: -len(SUFFIX)] != para:
        bad.append("question_instruction_mismatch")
    if jaccard(para, orig) >= JACCARD_MAX:
        bad.append("too_similar")
    low = ins[: -len(SUFFIX)].lower() if ins.endswith(SUFFIX) else ins.lower()
    # keyword search outside the expression so e.g. 'f(x)' cannot satisfy it by accident
    low_noexpr = low.replace(expr.lower(), " ") if expr else low
    kws = _KEYWORDS.get(kind)
    if kws is None:
        bad.append(f"unknown_kind:{kind}")
    elif not any(k in low_noexpr for k in kws):
        bad.append("operation_keyword_missing")
    if kind == "diff" and "integral" in low_noexpr:
        bad.append("wrong_operation_word")
    if kind in ("integral", "defint") and re.search(r"(?<!anti)derivative|(?<!anti)differentiat", low_noexpr):
        bad.append("wrong_operation_word")
    return bad


def verify(rows):
    kept, rejected = [], []
    for r in rows:
        bad = check_row(r)
        if bad:
            rejected.append({**r, "reject_reasons": bad})
        else:
            kept.append(r)
    return kept, rejected


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m lfm3u.audit.attacksets.paraphrase_math FILE.jsonl")
        return 2
    rows = [json.loads(line) for line in open(argv[0], encoding="utf-8") if line.strip()]
    kept, rejected = verify(rows)
    print(f"{argv[0]}: rows={len(rows)} kept={len(kept)} rejected={len(rejected)}")
    for r in rejected:
        print("  FAIL", r.get("id"), r["reject_reasons"])
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
