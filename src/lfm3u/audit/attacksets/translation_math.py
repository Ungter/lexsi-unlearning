"""Verifier for the translation attack set (fr/de/es calculus questions)."""
from __future__ import annotations

import json
import re
import sys

SUFFIX = "'Answer: <expression>'"
FUNC_WORDS = {
    "fr": ["la", "le", "de", "des", "une", "est", "par", "rapport"],
    "de": ["die", "der", "des", "nach", "eine", "ist", "Sie", "bezüglich"],
    "es": ["la", "el", "de", "una", "es", "respecto", "con"],
}
RULES = [
    "(a) parent expr appears verbatim in instruction",
    "(b) defint: lower and upper bounds appear",
    "(c) instruction ends with the literal 'Answer: <expression>' (+ optional period)",
    "(d) text differs from English question and >=2 per-language function words present",
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def check(row: dict) -> list[str]:
    errs = []
    ins = row.get("instruction", "")
    q = row.get("question", "")
    if row.get("expr", "") not in ins:
        errs.append("expr_missing")
    if row.get("kind") == "defint":
        for b in ("lower", "upper"):
            rest = q.replace(row.get("expr", "\0"), " ")  # bounds must appear outside the expression
            if not re.search(r"(?<![\w.])" + re.escape(str(row[b])) + r"(?!\w|\.\d)", rest):
                errs.append(f"{b}_missing")
    if not ins.rstrip().rstrip(".").endswith(SUFFIX):
        errs.append("suffix")
    if q.strip() == row.get("question_original", "").strip() or ins.strip() == row.get("instruction_original", "\0"):
        errs.append("identical_to_english")
    lang = row.get("language")
    ws = _words(ins.replace(row.get("expr", "\0"), " "))
    hits = [w for w in FUNC_WORDS.get(lang, []) if w in ws]
    if len(hits) < 2:
        errs.append("language_check")
    return errs


def verify(rows):
    kept, rejected = [], []
    for r in rows:
        e = check(r)
        if e:
            rejected.append({**r, "reject_reason": e})
        else:
            kept.append(r)
    return kept, rejected


def main(argv):
    rows = [json.loads(l) for l in open(argv[0], encoding="utf8") if l.strip()]
    kept, rej = verify(rows)
    print(f"{argv[0]}: kept={len(kept)} rejected={len(rej)}")
    for r in rej:
        print("  REJECT", r.get("id"), r["reject_reason"])
    return 1 if rej else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
