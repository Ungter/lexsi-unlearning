"""Algebra-aware helpers for the math attack-set verifiers (Tiny Aya branch: MATH_TARGET = algebra).

Algebra items from lfm3u.data.math_gen mostly have an empty `expr`, so the calculus checks ("parent expr verbatim",
"underlying_expression sympy-equal to expr") cannot see the math. This module extracts the math spans from the
templated question text and checks them instead:

  spans(row)            named math spans of the parent question (E, A/B for equations, E1/E2 for systems, R)
  paraphrase_errors()   spans verbatim, requested answer form still asked for, answer not leaked
  translation_errors()  spans verbatim
  word_errors()         story's underlying_expression is the parent's math (sympy: same expression, or an
                        equation that is a nonzero multiple of the parent's), requested form still asked for
"""
from __future__ import annotations

import re

import sympy as sp

from lfm3u.data.math_gen import same_expr

X, Y = sp.symbols("x y")

# (topic, kind, pattern on the ORIGINAL question). Groups: E = whole math span; A/B = equation sides; E1/E2 = system.
PATTERNS = [
    ("algebra", "solve", re.compile(r"^Solve the system: (?P<E1>[^,]+?), (?P<E2>[^,]+?) for x and y\. "
                                    r"Give the answer as an ordered pair \(x, y\)\.$")),
    ("algebra", "solve", re.compile(r"^Solve (?P<E>(?P<A>.+?) = (?P<B>.+?)) for x\.$")),
    ("algebra", "quadratic", re.compile(r"^Solve (?P<E>(?P<A>.+?) = (?P<B>0)) for x\. List all roots in increasing order\.$")),
    ("algebra", "factor", re.compile(r"^Factor (?P<E>.+?) over the integers\.$")),
    ("algebra", "algebra", re.compile(r"^Expand (?P<E>.+)\.$")),
    ("algebra", "simplify", re.compile(r"^Simplify (?P<E>.+?) for x != (?P<R>\S+)\.$")),
    ("arithmetic", "eval", re.compile(r"^Compute (?P<E>.+)\.$")),
    ("precalculus", "trig", re.compile(r"^Compute (?P<E>.+)\.$")),
    ("precalculus", "log", re.compile(r"^Evaluate (?P<E>.+)\.$")),
    ("precalculus", "identity", re.compile(r"^Simplify (?P<E>.+)\.$")),
    ("precalculus", "solve", re.compile(r"^Solve (?P<E>.+?) (?P<C>where .+|for real x)\.$")),
]

# words that keep the requested ANSWER FORM (math_gen.check_answer enforces the form for factor/expand/simplify)
PARA_FORM = {
    "quadratic": r"increasing|ascending|smallest to largest|lowest to highest|least to greatest|smaller (?:root|one|value) first",
    "system": r"ordered pair|\(x, y\)",
    "factor": r"integer",
}
WORD_FORM = {
    "factor": r"factor",
    "algebra": r"expand|multipl(?:y|ied) out|standard form|without (?:any )?parenthes",
    "simplify": r"simplif|polynomial",
    "quadratic": PARA_FORM["quadratic"],
    "system": PARA_FORM["system"],
}


def spans(row: dict, field: str = "question_original") -> dict:
    """Named math spans of the parent question ({} if the question does not match a known template)."""
    q = (row.get(field) or row.get("question") or "").strip()
    for topic, kind, pat in PATTERNS:
        if row.get("topic") == topic and row.get("kind") == kind:
            m = pat.match(q)
            if m:
                return {k: v for k, v in m.groupdict().items() if v is not None}
    if row.get("expr"):
        out = {"E": row["expr"]}
        if row.get("kind") == "defint":
            out.update(L=str(row.get("lower", "")), U=str(row.get("upper", "")))
        return out
    return {}


def is_system(row: dict) -> bool:
    return "E1" in spans(row)


def fill(template: str, row: dict, var: str = "x") -> str:
    """Replace {E} {A} {B} {E1} {E2} {R} {L} {U} {C} with the parent's spans (x renamed to var)."""
    sp_ = spans(row)

    def rep(m):
        v = sp_[m.group(1)]
        return rename(v, "x", var) if m.group(1) not in ("R", "L", "U") else v
    return re.sub(r"\{(E1|E2|E|A|B|R|L|U|C)\}", rep, template)


def rename(text: str, old: str, new: str) -> str:
    if old == new:
        return text
    return re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(old), new, text)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).replace("**", "^").replace("\u2212", "-")


def _leak(text: str, row: dict, blank: list[str]) -> bool:
    rest = _norm(text)
    for b in sorted(blank, key=len, reverse=True):
        rest = rest.replace(_norm(b), "#")
    ans = _norm(row.get("answer", ""))
    return len(ans) >= 3 and ans in rest


def _form_key(row: dict) -> str:
    return "system" if is_system(row) else row.get("kind", "")


def paraphrase_errors(row: dict, text: str) -> list[str]:
    bad = []
    s = spans(row)
    if not s:
        return ["algebra_template_unrecognised"]
    sides = "A" in s and s["A"] in text and s["B"] in text  # 'A equals B' keeps the equation too
    for k, v in s.items():
        if k in ("A", "B") or (k == "E" and sides):
            continue  # covered by E (or E by A and B)
        if k == "R":
            if not _restriction_kept(s, text):
                bad.append("restriction_missing")
        elif v not in text:
            bad.append(f"span_not_verbatim:{k}")
    pat = PARA_FORM.get(_form_key(row))
    if pat and not re.search(pat, text, flags=re.I):
        bad.append("answer_form_missing")
    if _leak(text, row, list(s.values())):
        bad.append("answer_leaked")
    return bad


def translation_errors(row: dict, text: str) -> list[str]:
    s = spans(row)
    if not s:
        return ["algebra_template_unrecognised"]
    sides = "A" in s and s["A"] in text and s["B"] in text
    bad = [f"span_not_verbatim:{k}" for k, v in s.items()
           if k not in ("A", "B", "R") and not (k == "E" and sides) and v not in text]
    if "R" in s and not _restriction_kept(s, text):
        bad.append("restriction_missing")
    return bad


def _restriction_kept(s: dict, text: str) -> bool:
    """The x != r value must appear OUTSIDE the fraction (the divisor x - (r) always contains it)."""
    rest = text.replace(s.get("E", "\0"), " ")
    return re.search(r"(?<![\w.])" + re.escape(s["R"]) + r"(?![\w]|\.\d)", rest) is not None


_SAFE = re.compile(r"^[0-9xy\s+\-*/().=,^]+$")


def _sym(text: str):
    text = text.strip()
    if not _SAFE.match(text):
        raise ValueError(f"unexpected characters in {text!r}")
    return sp.sympify(text.replace("^", "**"), locals={"x": X, "y": Y})


def _eq(text: str):
    """'lhs = rhs' -> lhs - rhs (a plain expression is returned as is)."""
    parts = text.split("=")
    if len(parts) == 1:
        return _sym(parts[0])
    if len(parts) != 2:
        raise ValueError(f"not one equation: {text!r}")
    return sp.expand(_sym(parts[0]) - _sym(parts[1]))


def _proportional(a, b) -> bool:
    if b == 0 or a == 0:
        return False
    r = sp.simplify(a / b)
    return bool(r.is_number and r != 0)


def word_errors(row: dict) -> str | None:
    """Algebra path of word_problems._check: underlying_expression must be the parent's math."""
    var = row["variable"]
    ue = rename(row.get("underlying_expression", ""), var, "x")
    s = spans(row)
    if not s:
        return "algebra_template_unrecognised"
    try:
        if "E1" in s:
            if var != "x":
                return "system word problems must use the variables x and y"
            got = [_eq(e) for e in ue.split(",")]
            want = [_eq(s["E1"]), _eq(s["E2"])]
            if len(got) != 2 or not ((_proportional(got[0], want[0]) and _proportional(got[1], want[1]))
                                     or (_proportional(got[0], want[1]) and _proportional(got[1], want[0]))):
                return "underlying system != parent system"
        elif "A" in s:  # linear equation or quadratic = 0
            if ue.count("=") != 1 and row.get("kind") != "quadratic":
                return "underlying_expression is not an equation"
            if not _proportional(_eq(ue), _eq(s["E"])):
                return "underlying equation != parent equation"
        else:  # factor / expand / simplify: the expression itself
            if not same_expr(ue, s["E"]):
                return "underlying_expression != parent expression"
    except Exception as exc:  # noqa: BLE001
        return "underlying_expression unparsable: %s" % exc
    pat = WORD_FORM.get(_form_key(row))
    if pat and not re.search(pat, row.get("question", ""), flags=re.I):
        return "requested answer form missing from text"
    return None
