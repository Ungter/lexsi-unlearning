"""Phase-1 go/no-go calculus probe set (results/base/baseline.json was frozen with it).

Separate from lfm3u.data.math_gen, which builds the Phase-2 train/dev/test splits.
Sympy problems: derivatives, indefinite and definite integrals at levels 1-3.

Every problem carries a machine-checkable reference (sympy srepr) used by lfm3u.grade:
  diff  -> check "equal"      ref = f'(x)
  indef -> check "antideriv"  ref = integrand (prediction is differentiated, so the constant doesn't matter)
  def   -> check "number"     ref = exact value
"""
from __future__ import annotations

import random
import re

import sympy as sp

x = sp.Symbol("x")

KINDS = (("diff", 0.6), ("indef", 0.25), ("def", 0.15))
MAX_OPS = {1: 30, 2: 45, 3: 70}
SUFFIX = "Give the final answer on the last line as 'Answer: <expression>'."


def show(e) -> str:
    """Plain-text rendering used in prompts: ^ for powers, e for Euler's number."""
    s = sp.sstr(e).replace("**", "^")
    return re.sub(r"\bE\b", "e", s)


def _k(r: random.Random, lo: int = 1, hi: int = 9, neg: bool = True) -> int:
    v = r.randint(lo, hi)
    return -v if neg and r.random() < 0.3 else v


def _poly(r: random.Random, deg: int):
    ks = {deg} | set(r.sample(range(deg), k=min(deg, r.randint(1, 2))))
    return sp.Add(*[_k(r) * x**k for k in ks])


def _atom(r: random.Random):
    """Term that needs a single basic rule."""
    k, a = _k(r, 1, 7), r.randint(1, 5)
    return r.choice([
        lambda: k * x ** r.randint(2, 7),
        lambda: k * sp.sin(a * x),
        lambda: k * sp.cos(a * x),
        lambda: k * sp.exp(a * x),
        lambda: k * sp.log(x),
        lambda: k * sp.sqrt(x),
        lambda: k / x ** r.randint(1, 3),
    ])()


_FUNCS = ("pow", "sin", "cos", "exp", "log", "sqrt")


def _func(r: random.Random, kind: str | None = None):
    a = r.randint(1, 4)
    kind = kind or r.choice(_FUNCS)
    return {
        "pow": lambda: x ** r.randint(1, 4),
        "sin": lambda: sp.sin(a * x),
        "cos": lambda: sp.cos(a * x),
        "exp": lambda: sp.exp(a * x),
        "log": lambda: sp.log(x),
        "sqrt": lambda: sp.sqrt(x),
    }[kind]()


def _two_funcs(r: random.Random):
    a, b = r.sample(_FUNCS, 2)
    return _func(r, a), _func(r, b)


def _inner(r: random.Random):
    """Inner function for the chain rule and whether it is positive for x > 0."""
    a, b, c = r.randint(1, 5), _k(r, 1, 7), r.randint(1, 9)
    return r.choice([(a * x + b, False), (x**2 + c, True), (a * x**2 + b * x, False), (x**3 + c, True), (a * x**2 + c, True)])


def _outer(r: random.Random, u, positive: bool):
    opts = [sp.sin(u), sp.cos(u), sp.exp(u), u ** r.randint(2, 5)]
    if positive:
        opts += [sp.log(u), sp.sqrt(u)]
    return r.choice(opts)


def _composed(r: random.Random):
    return _outer(r, *_inner(r))


def _is_monomial(f) -> bool:
    return bool(f.is_polynomial(x)) or (f.is_Pow and f.base == x) or (
        f.is_Mul and all(a.is_number or a == x or (a.is_Pow and a.base == x) for a in f.args))


def _diff_problem(r: random.Random, level: int):
    if level == 1:
        f = r.choice([lambda: _atom(r) + _atom(r), lambda: _poly(r, r.randint(3, 6)), lambda: _atom(r) + _atom(r) + r.randint(1, 9)])()
    elif level == 2:
        f = r.choice([
            lambda: sp.Mul(*_two_funcs(r)),
            lambda: _func(r) / _poly(r, r.randint(1, 2)),
            lambda: _k(r, 1, 5) * _composed(r),
        ])()
    else:
        f = r.choice([
            lambda: _composed(r) * _func(r),
            lambda: _composed(r) / _poly(r, r.randint(1, 2)),
            lambda: _outer(r, _composed(r), False),
            lambda: sp.exp(sp.Mul(*_two_funcs(r))),
            lambda: sp.sin(sp.Mul(*_two_funcs(r))),
        ])()
    if level >= 2 and _is_monomial(f):
        return None
    ref = sp.diff(f, x)
    prompt = f"Find the derivative of f(x) = {show(f)} with respect to x."
    return f, ref, prompt, {"check": "equal", "answer": ref}


_PARTS2 = (lambda a: x * sp.exp(a * x), lambda a: x * sp.sin(a * x), lambda a: x * sp.cos(a * x),
           lambda a: sp.log(x), lambda a: x * sp.log(x))
_TEMPL3 = (lambda a, b: x**2 * sp.exp(a * x), lambda a, b: x**2 * sp.sin(a * x), lambda a, b: x**2 * sp.cos(a * x),
           lambda a, b: sp.exp(a * x) * sp.sin(b * x), lambda a, b: sp.exp(a * x) * sp.cos(b * x),
           lambda a, b: 1 / ((x + a) * (x + a + b)), lambda a, b: x**2 * sp.log(x), lambda a, b: sp.log(x) ** 2,
           lambda a, b: x / (x**2 + a), lambda a, b: x**3 * sp.exp(x**2))


def _indef_problem(r: random.Random, level: int):
    a, b = r.randint(1, 4), r.randint(1, 4)
    if level == 1:
        F = _atom(r) + _atom(r)
        g = sp.diff(F, x)
    elif level == 2:
        if r.random() < 0.5:
            F = _k(r, 1, 5) * _composed(r)
            g = sp.diff(F, x)
        else:
            g = r.choice(_PARTS2)(a)
            F = sp.integrate(g, x)
    else:
        if r.random() < 0.3:
            F = sp.exp(sp.Mul(*_two_funcs(r)))
            g = sp.diff(F, x)
        else:
            g = r.choice(_TEMPL3)(a, b)
            F = sp.integrate(g, x)
    if F.has(sp.Integral) or g == 0:
        return None
    prompt = f"Find the indefinite integral of f(x) = {show(g)} with respect to x. You may omit the constant of integration."
    return g, g, prompt, {"check": "antideriv", "answer": F}


def _def_problem(r: random.Random, level: int):
    if level == 1:
        g = _poly(r, r.randint(1, 2))
        lo = r.randint(-2, 1)
        hi = lo + r.randint(1, 3)
    elif level == 2:
        k = _k(r, 1, 6)
        g, lo, hi = r.choice([
            lambda: (_poly(r, 3), 0, r.randint(1, 3)),
            lambda: (k * sp.sin(x), 0, r.choice([sp.pi / 2, sp.pi])),
            lambda: (k * sp.cos(x), 0, sp.pi / 2),
            lambda: (k * sp.exp(x), 0, 1),
            lambda: (k / x, 1, sp.E),
        ])()
    else:
        g, lo, hi = r.choice([
            (x * sp.exp(x), 0, 1), (x * sp.sin(x), 0, sp.pi), (sp.log(x), 1, sp.E), (x * sp.cos(x), 0, sp.pi / 2),
            (x / (x**2 + 1), 0, 1), (x * sp.exp(x**2), 0, 1), (x**2 * sp.exp(x), 0, 1), (sp.sin(x) ** 2, 0, sp.pi),
            (x**2 * sp.sin(x), 0, sp.pi), (sp.exp(x) * sp.sin(x), 0, sp.pi), (x * sp.log(x), 1, sp.E),
            (1 / (x * (x + 1)), 1, 2), (x * sp.sqrt(x**2 + 1), 0, 1), (sp.cos(x) ** 2, 0, sp.pi / 2),
        ])
        g = r.randint(1, 5) * g
    val = sp.simplify(sp.integrate(g, (x, lo, hi)))
    if val.has(sp.Integral) or not val.is_number:
        return None
    prompt = (f"Evaluate the definite integral of f(x) = {show(g)} with respect to x from x = {show(lo)} to x = {show(hi)}. "
              "Give the exact value.")
    return g, val, prompt, {"check": "number", "answer": val, "bounds": [show(lo), show(hi)]}


_MAKERS = {"diff": _diff_problem, "indef": _indef_problem, "def": _def_problem}


def generate_calculus(n_per_level: int = 70, levels=(1, 2, 3), seed: int = 0) -> list[dict]:
    """Deterministic list of calculus problems (60% diff, 25% indefinite, 15% definite per level)."""
    out, seen = [], set()
    for level in levels:
        counts = [round(n_per_level * w) for _, w in KINDS]
        counts[0] += n_per_level - sum(counts)
        for ki, ((kind, _), n) in enumerate(zip(KINDS, counts)):
            r = random.Random(seed * 1000 + level * 10 + ki)
            made, tries = 0, 0
            while made < n:
                tries += 1
                if tries > 200 * n:
                    raise RuntimeError(f"could not generate enough {kind} problems at level {level}")
                res = _MAKERS[kind](r, level)
                if res is None:
                    continue
                f, ref, prompt, meta = res
                if ref == 0 or ref.has(sp.nan, sp.zoo, sp.oo) or sp.count_ops(ref) > MAX_OPS[level]:
                    continue
                if "log(" in prompt:
                    prompt += " Here log denotes the natural logarithm."
                prompt = f"{prompt} {SUFFIX}"
                if prompt in seen:
                    continue
                seen.add(prompt)
                out.append({
                    "id": f"calc-L{level}-{kind}-{made:03d}",
                    "target": "calculus",
                    "level": level,
                    "kind": kind,
                    "prompt": prompt,
                    "func": show(f),
                    "answer": show(meta["answer"]),
                    "check": meta["check"],
                    "ref_srepr": sp.srepr(ref),
                    **({"bounds": meta["bounds"]} if "bounds" in meta else {}),
                })
                made += 1
    return out
