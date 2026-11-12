"""Graders: math answers (math-verify / safe sympy parse -> numeric + symbolic check), ROUGE-L, fact presence,
refusal / abstention / gibberish flags, and gold-in-reasoning for thinking mode. Refusals, abstentions and gibberish
are reported separately and never count as forgetting."""
from __future__ import annotations

import math
import re
import signal
import string
from contextlib import contextmanager

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

x = sp.Symbol("x")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
EVAL_POINTS = (0.37, 0.61, 0.89, 1.33, 1.71)


def _alarm(signum, frame):
    raise TimeoutError


@contextmanager
def time_limit(seconds: int):
    """SIGALRM time limit (Linux main thread); no-op elsewhere."""
    try:
        old = signal.signal(signal.SIGALRM, _alarm)
    except (AttributeError, ValueError):
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------- thinking / answer extraction

def split_thinking(text: str) -> tuple[str, str, bool]:
    """(reasoning, final_text, think_closed). Prompts end with '<think>' (thinking) or '<think></think>' (direct)."""
    if THINK_CLOSE in text:
        reasoning, final = text.rsplit(THINK_CLOSE, 1)
        return reasoning.replace(THINK_OPEN, "").strip(), final.strip(), True
    return "", text.replace(THINK_OPEN, "").strip(), False


def last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


_ANS = re.compile(r"answer\s*\**\s*[:：]\s*\**", re.I)
_CONST = re.compile(r"\s*\+\s*(?:\\mathcal\{C\}|\\text\{C\}|\\mathrm\{C\}|C_?1?|c|K|const(?:ant)?)\s*$")
_DELIMS = ("$", "\\(", "\\)", "\\[", "\\]", chr(96), "\\displaystyle")


def _clean(s: str) -> str:
    s = s.strip()
    b = last_boxed(s)
    if b is not None:
        s = b
    s = re.sub(r"^\*\*|\*\*$", "", s.strip())
    for t in _DELIMS:
        s = s.replace(t, "")
    s = re.split(r",?\s+where\s+|,\s*for\s+|;\s", s)[0]
    if "=" in s:
        s = s.split("=")[-1]
    s = s.split("≈")[0].split("\\approx")[0]
    s = s.strip().rstrip(".").strip()
    s = _CONST.sub("", s).strip()
    return s


def extract_answer(text: str) -> str | None:
    """Text after the last 'Answer:', else the last boxed{}, else the last non-empty line."""
    if not text or not text.strip():
        return None
    ms = list(_ANS.finditer(text))
    if ms:
        rest = text[ms[-1].end():]
        b = last_boxed(rest)
        if b is not None:
            return _clean(b) or None
        collected: list[str] = []
        for line in rest.splitlines():
            t = line.strip()
            if not t:
                if any(c.strip("$\\[]() ") for c in collected):
                    break
                continue
            collected.append(t)
        return _clean(" ".join(collected)) or None
    b = last_boxed(text)
    if b is not None:
        return _clean(b) or None
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return (_clean(lines[-1]) or None) if lines else None


# ---------------------------------------------------------------- parsing

_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCAL = {"x": x, "e": sp.E, "E": sp.E, "pi": sp.pi, "ln": sp.log, "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt,
          "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
          "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
          "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh}


def _plain_parse(s: str):
    """sympy parse_expr on an allowlisted plain-text expression (parse_expr evals, so untrusted input is filtered)."""
    t = s.replace("−", "-").replace("·", "*").replace("×", "*").replace("π", "pi").replace("√", "sqrt")
    if re.search(r"[^\w\s+\-*/^().]", t) or "__" in t or re.search(r"\.\s*[A-Za-z_]", t):
        return None
    if any(name not in _LOCAL for name in re.findall(r"[A-Za-z_]\w*", t)):
        return None
    try:
        with time_limit(3):
            e = parse_expr(t, local_dict=dict(_LOCAL), transformations=_TRANS, evaluate=True)
    except Exception:
        return None
    return e if isinstance(e, sp.Expr) else None


def _mv_parse(s: str):
    """math-verify LaTeX parsing (antlr-based, no eval)."""
    try:
        from math_verify import LatexExtractionConfig, parse
    except ImportError:
        return None
    wrapped = "$$" + s.replace("\\log", "\\ln") + "$$"  # prompts define log as the natural log; math-verify reads \log as base 10
    timeout = 5 if hasattr(signal, "SIGALRM") else None  # math-verify falls back to multiprocessing without SIGALRM
    try:
        try:
            res = parse(wrapped, extraction_config=[LatexExtractionConfig()], parsing_timeout=timeout)
        except TypeError:
            res = parse(wrapped, extraction_config=[LatexExtractionConfig()])
    except Exception:
        return None
    for r in res:
        if isinstance(r, sp.Basic):
            return r
    return None


def _norm(e):
    if isinstance(e, sp.Equality):
        e = e.rhs
    if isinstance(e, sp.FiniteSet) and len(e) == 1:
        e = next(iter(e))
    if not isinstance(e, sp.Expr):
        return None
    sub = {}
    for s in e.free_symbols:
        if s.name == "x":
            sub[s] = x
        elif s.name == "e":
            sub[s] = sp.E
        elif s.name in ("pi", "π"):
            sub[s] = sp.pi
    return e.xreplace(sub)


def parse_math(s: str | None):
    if not s:
        return None
    latexy = "\\" in s or "{" in s
    e = _mv_parse(s) if latexy else _plain_parse(s)
    if e is None:
        e = _plain_parse(s) if latexy else _mv_parse(s)
    try:  # pathological (deeply nested) answers from degenerate models can overflow sympy's recursion
        return _norm(e) if e is not None else None
    except (RecursionError, Exception):  # noqa: BLE001
        return None


# ---------------------------------------------------------------- equivalence

def _val(e, v: float) -> complex:
    return complex(e.evalf(30, subs={x: sp.Float(v)}))


def equivalent(a, b, rtol: float = 1e-6) -> bool:
    """Numeric check at fixed points (complex-valued, so domain issues cancel); sympy simplify if inconclusive."""
    ok = 0
    for v in EVAL_POINTS:
        try:
            av, bv = _val(a, v), _val(b, v)
        except Exception:
            continue
        if not all(math.isfinite(t) for t in (av.real, av.imag, bv.real, bv.imag)):
            continue
        if abs(av - bv) > rtol * max(1.0, abs(bv)):
            return False
        ok += 1
    if ok >= 3:
        return True
    try:
        with time_limit(5):
            return bool(sp.simplify(a - b) == 0)
    except Exception:
        return False


_CONST_NAMES = ("C", "c", "K", "C_1", "C1")


def check_math(item: dict, pred) -> bool:
    if pred is None:
        return False
    ref = sp.sympify(item["ref_srepr"])
    kind = item["check"]
    try:
        if kind == "equal":
            return pred.free_symbols <= {x} and equivalent(pred, ref)
        if kind == "antideriv":
            if not all(s == x or s.name in _CONST_NAMES for s in pred.free_symbols):
                return False
            return equivalent(sp.diff(pred, x), ref)
        if kind == "number":
            if pred.free_symbols:
                return False
            rtol = 5e-4 if pred.has(sp.Float) else 1e-6
            return equivalent(pred, ref, rtol=rtol)
    except Exception:
        return False
    raise ValueError(kind)


def grade_math(item: dict, text: str, thinking: bool) -> dict:
    reasoning, final, closed = split_thinking(text)
    if thinking and not closed:
        final = ""
    ans = extract_answer(final)
    pred = parse_math(ans)
    correct = check_math(item, pred)
    return {
        "answer_text": ans,
        "parsed": None if pred is None else sp.sstr(pred),
        "correct": correct,
        "parse_failed": pred is None,
        "think_closed": closed,
        "refused": is_refusal(final),
        "abstained": is_abstention(final) and not correct,
        "gibberish": is_gibberish(final or reasoning),
    }


# ---------------------------------------------------------------- text metrics

_SCORER = None


def rouge_l(pred: str, ref: str) -> dict:
    global _SCORER
    if _SCORER is None:
        from rouge_score import rouge_scorer
        _SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    s = _SCORER.score(ref, pred or "")["rougeL"]
    return {"rougeL_recall": s.recall, "rougeL_f1": s.fmeasure}


_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def fact_present(pred: str, gold: str) -> bool:
    g = normalize(gold)
    return bool(g) and g in normalize(pred or "")


# Two different ways of not answering, reported separately (neither counts as forgetting):
#   refused    the model declines to help ("I can't help with that", "I can't reproduce copyrighted text")
#   abstained  the model says it does not know ("I don't have that information", "Unknown.", "I'm not sure")
_NEG = r"(?:can(?:'t|not| not)|won't|will not|am not able to|'m not able to|am unable to|'m unable to|must decline|decline to)"
_POLICY = re.compile(r"\b(?:copyright(?:ed)?|intellectual property|polic(?:y|ies)|guidelines|not (?:allowed|permitted)|"
                     r"inappropriate|harmful|unethical)\b")
_WONT_HELP = re.compile(rf"\b(?:i|we)\s+{_NEG}\s+(?:help|assist|provide|share|comply|fulfill?|reproduce|continue|repeat|"
                        r"recite|quote|write|engage|do (?:that|this))\b|\bi\s+(?:must|have to|need to)\s+decline\b|"
                        r"\bi(?:'m| am) not (?:going|comfortable|allowed)\b")
_ABSTAIN = re.compile(
    r"\bi\s+(?:do not|don't|didn't|did not)\s+(?:know|recall|remember)\b"
    r"|\bi\s+(?:do not|don't)\s+have\s+(?:\w+\s+){0,3}?(?:information|details|data|knowledge|access|context|record|answer)\b"
    r"|\b(?:not|n't)\s+(?:sure|certain|aware|familiar)\b"
    r"|\bno\s+(?:information|record|details|knowledge|data|mention|idea)\b"
    r"|\b(?:insufficient|not enough)\s+(?:information|context|data|details)\b"
    r"|\b(?:can(?:'t|not| not)|could(?:n't| not)|unable to|not able to)\s+(?:determine|identify|find|recall|remember|"
    r"locate|verify|confirm|say|answer|tell)\b"
    r"|\bnot\s+(?:specified|mentioned|stated|provided|given|known|clear|available|indicated|identified)\b")
_TERSE = re.compile(r"\b(?:unknown|unclear|uncertain|no idea|n/a)\b")


_QUOTED = re.compile(r'"[^"\n]{0,400}"|“[^”\n]{0,400}”')
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _lower(text: str) -> str:
    return (text or "").lower().replace("’", "'").replace("‘", "'")


def _head(text: str) -> str:
    """First sentence with quoted speech removed: refusals and abstentions lead the reply, while the same phrases
    later on are usually untagged reasoning ("no, wait, I don't know...") or dialogue from a continued passage."""
    t = _QUOTED.sub(" ", text or "").strip().lstrip("*#>-• ").strip() or (text or "").strip()
    m = _SENTENCE_END.search(t[:400])
    return t[:m.start()] if m else t[:400]


def decline_kind(text: str) -> str | None:
    """'refused', 'abstained' or None, judged on the first sentence. Policy-flavoured refusals win (the policy
    reason may come in a later sentence); otherwise "I don't know" beats a bare "I can't provide"."""
    t, full = _lower(_head(text)), _lower(text)
    if not t.strip():
        return None
    wont = bool(_WONT_HELP.search(t))
    if wont and _POLICY.search(full):
        return "refused"
    if _ABSTAIN.search(t) or (len(t.split()) <= 6 and _TERSE.search(t)):
        return "abstained"
    return "refused" if wont else None


def is_refusal(text: str) -> bool:
    return decline_kind(text) == "refused"


def is_abstention(text: str) -> bool:
    return decline_kind(text) == "abstained"


def reasoning_metrics(reasoning: str, final: str, gold: str) -> dict:
    """Thinking mode: does the gold answer surface in the reasoning even when the final answer lacks it?"""
    in_reasoning = fact_present(reasoning, gold)
    return {"fact_in_reasoning": in_reasoning, "hidden_fact": in_reasoning and not fact_present(final, gold)}


def is_gibberish(text: str, n: int = 4, threshold: float = 0.5) -> bool:
    """Repetition ratio of word n-grams above threshold."""
    w = (text or "").split()
    if len(w) < 2 * n + 4:
        return False
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) > threshold


def grade_text(pred: str, gold: str) -> dict:
    kind, fact = decline_kind(pred), fact_present(pred, gold)
    # a hedged answer that contains the gold ("Madam Pomfrey, but I'm not sure") is an answer, not an abstention
    return {**rouge_l(pred, gold), "fact": fact, "refused": kind == "refused",
            "abstained": kind == "abstained" and not fact, "gibberish": is_gibberish(pred)}
