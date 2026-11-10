"""Grade model outputs on lfm3u.data.math_gen items (Phase-2 splits and the attack sets built from them).

Calculus kinds (diff / integral / defint) are checked numerically with lfm3u.grade (derivative compared to f' for
diff, derivative of the prediction compared to the integrand for integral, value for defint). Other topics use
math_gen.check_answer (answer lists, form checks), first on the extracted text and then on its LaTeX-parsed form.
Rows with a "variable" field (word problems) have that variable mapped back to x before grading.
"""
from __future__ import annotations

import re

import sympy as sp

from lfm3u import grade
from lfm3u.data import math_gen as mg

CALC_CHECK = {"diff": "equal", "integral": "antideriv", "defint": "number"}


def _calc_correct(row: dict, pred) -> bool:
    if pred is None:
        return False
    kind = row["kind"]
    try:
        ref = mg.parse_expression(row["expr"] if kind == "integral" else row["answer"])
    except Exception:  # noqa: BLE001
        return False
    return grade.check_math({"ref_srepr": sp.srepr(ref), "check": CALC_CHECK[kind]}, pred)


def _other_correct(row: dict, ans: str) -> bool:
    try:
        with grade.time_limit(5):
            if mg.check_answer(row, ans):
                return True
    except Exception:  # noqa: BLE001
        pass
    pred = grade.parse_math(ans)
    if pred is None:
        return False
    try:
        with grade.time_limit(5):
            return bool(mg.check_answer(row, sp.sstr(pred)))
    except Exception:  # noqa: BLE001
        return False


def grade_item(row: dict, text: str, thinking: bool = False) -> dict:
    reasoning, final, closed = grade.split_thinking(text)
    if thinking and not closed:
        final = ""
    ans = grade.extract_answer(final)
    var = row.get("variable")
    if ans and var and var != "x":
        ans = re.sub(rf"\b{re.escape(var)}\b", "x", ans)
    if not ans:
        correct, parsed = False, None
    elif row.get("kind") in CALC_CHECK:
        pred = grade.parse_math(ans)
        correct, parsed = _calc_correct(row, pred), (None if pred is None else sp.sstr(pred))
    else:
        correct, parsed = _other_correct(row, ans), None
    kind = grade.decline_kind(final)
    return {"answer_text": ans, "parsed": parsed, "correct": bool(correct), "think_closed": closed,
            "refused": kind == "refused", "abstained": kind == "abstained" and not correct,
            "gibberish": grade.is_gibberish(final or reasoning)}
