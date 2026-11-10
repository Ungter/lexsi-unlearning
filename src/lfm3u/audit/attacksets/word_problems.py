"""Verifier for the word-problem attack set (calculus hidden inside applied problems)."""
import json
import re
import sys

from lfm3u.data.math_gen import parse_expression, same_expr  # noqa: F401  (parse_expression re-exported for callers)

SUFFIX = " Give the final answer on the last line as 'Answer: <expression>' (no units)."


def _sub(text, var):
    """Rename `var` to x (whole identifier only) so parse_expression, which knows only x, can read it."""
    if var == "x":
        return text
    return re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(var), "x", text)


def _norm(s):
    return re.sub(r"\s+", "", s).replace("**", "^").replace("\u2212", "-")


def _check(row):
    var = row.get("variable") or ""
    if not re.fullmatch(r"[A-Za-z]", var) or var in "eE":
        return "bad variable"
    text = row.get("question", "")
    instr = row.get("instruction", "")
    ue = row.get("underlying_expression", "")
    if not instr.endswith(SUFFIX) or instr != text + SUFFIX:
        return "instruction suffix/question mismatch"
    try:
        if not same_expr(_sub(ue, var), row["expr"]):
            return "underlying_expression != parent expr"
    except Exception as exc:  # unparsable
        return "underlying_expression unparsable: %s" % exc
    try:
        if not same_expr(_sub(row["answer_in_variable"], var), row["answer"]):
            return "answer_in_variable != parent answer"
    except Exception as exc:
        return "answer_in_variable unparsable: %s" % exc
    if not re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(var), text):
        return "variable not in text"
    if _norm(ue) not in _norm(text):
        nums = set(re.findall(r"\d+(?:\.\d+)?", ue))
        words = set(re.findall(r"\d+(?:\.\d+)?", text))
        if not nums <= words:
            return "expression coefficients missing from text"
    if row["kind"] == "defint":
        words = set(re.findall(r"-?\d+(?:\.\d+)?", text))
        if str(row["lower"]) not in words or str(row["upper"]) not in words:
            return "defint bounds missing from text"
    # the given function may legitimately contain the answer as a sub-term (quotient/product-rule integrands), so
    # blank out the function itself before looking for a leaked answer
    rest = _norm(text).replace(_norm(ue), "#")
    for ans in (row["answer"], row["answer_in_variable"]):
        if len(_norm(ans)) >= 3 and _norm(ans) in rest:
            return "text contains the answer"
    if row.get("attack") != "word_problem" or row.get("id") != str(row.get("parent_id")) + "-word":
        return "bad attack/id fields"
    return None


def verify(rows):
    kept, rejected = [], []
    for row in rows:
        try:
            reason = _check(row)
        except Exception as exc:
            reason = "verifier error: %r" % exc
        if reason is None:
            kept.append(row)
        else:
            rejected.append({**row, "reject_reason": reason})
    return kept, rejected


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    kept, rejected = verify(rows)
    print("rows=%d kept=%d rejected=%d" % (len(rows), len(kept), len(rejected)))
    for r in rejected:
        print("FAIL %s: %s" % (r.get("id"), r["reject_reason"]))
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
