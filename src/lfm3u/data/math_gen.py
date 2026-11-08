import ast
import hashlib
import random
import re
from dataclasses import dataclass

import sympy as sp

from .splits import (
    assert_disjoint,
    ensure_writable,
    group_split,
    source_id,
    split_counts,
    write_jsonl,
)

X = sp.Symbol("x")
LOCALS = {"x": X, "pi": sp.pi, "E": sp.E, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
          "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs, "Rational": sp.Rational}
SUFFIX = " Give the final answer on the last line as 'Answer: <expression>'."
TOPICS = ("arithmetic", "algebra", "precalculus", "calculus")
LEVELS = (1, 2, 3)
SIZES = {"train": 1500, "dev": 300, "test": 500}
TRIG = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan}
ANGLES = (sp.pi / 6, sp.pi / 4, sp.pi / 3, sp.pi / 2, 2 * sp.pi / 3, 3 * sp.pi / 4, 5 * sp.pi / 6, sp.pi)
TRIG_BOUNDS = (sp.Integer(0), sp.pi / 2, sp.pi)
POLY_BOUNDS = (sp.Integer(0), sp.Integer(1), sp.Integer(2), sp.Integer(3))
TRIG_SOLUTIONS = {"1/2": "pi/6", "sqrt(2)/2": "pi/4", "sqrt(3)/2": "pi/3"}
HENDRYCKS_LEVEL = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}


@dataclass(frozen=True)
class MathItem:
    id: str
    topic: str
    level: int
    kind: str
    question: str
    answer: str
    expr: str = ""
    source_id: str = ""
    primitive: str = ""
    lower: str = ""
    upper: str = ""

    def to_alpaca(self):
        return {
            "instruction": self.question + SUFFIX,
            "output": f"Answer: {self.answer}",
            "id": self.id,
            "source_id": self.source_id or self.id,
            "topic": self.topic,
            "level": self.level,
            "kind": self.kind,
            "question": self.question,
            "answer": self.answer,
            "expr": self.expr,
            "primitive": self.primitive,
            "lower": self.lower,
            "upper": self.upper,
        }


def parse_expression(text):
    text = str(text).strip()
    if len(text) > 4096:
        raise ValueError("expression is too long")
    tree = ast.parse(text.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("expression is too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 10 ** 100:
                raise ValueError("numeric constant is too large")
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in ("x", "pi", "E"):
            return LOCALS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.is_number and (not right.is_finite or abs(right) > 100):
                    raise ValueError("exponent is too large")
                return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in LOCALS and callable(LOCALS[node.func.id]) and not node.keywords):
            if not 1 <= len(node.args) <= 2:
                raise ValueError("invalid function arguments")
            return LOCALS[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError("unsupported mathematical syntax")

    result = visit(tree.body)
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("expression is not finite")
    return result


def same_expr(a, b):
    try:
        diff = sp.simplify(parse_expression(a) - parse_expression(b))
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
        return False
    return diff == 0


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j].strip()
    return None


def _mag(rng, lo, hi):
    return rng.randint(lo, hi)


def _sgn(rng, lo, hi):
    return rng.randint(lo, hi) * rng.choice((-1, 1))


def _plus(v):
    return f" + {v}" if v >= 0 else f" - {-v}"


def _poly(rng, lo, hi):
    d = rng.randint(lo, hi)
    return sp.Add(*[rng.randint(-9, 9) * X ** k for k in range(d)], _sgn(rng, 1, 9) * X ** d)


def _atom(rng):
    a, b, m = _mag(rng, 1, 30), _mag(rng, 1, 12), _mag(rng, 1, 30)
    kind = rng.randrange(5)
    if kind == 0:
        return _poly(rng, 2, 4)
    if kind == 1:
        return a * sp.sin(b * X + m)
    if kind == 2:
        return a * sp.cos(b * X + m)
    if kind == 3:
        return a * sp.exp(b * X) + m
    return a * sp.log(b * X + m)


def _fn(rng, level):
    a, b, c, m = _mag(rng, 1, 20), _mag(rng, 1, 9), _mag(rng, 1, 20), _mag(rng, 1, 20)
    if level == 1:
        return _atom(rng)
    if level == 2:
        k = rng.randrange(4)
        if k == 0:
            return _atom(rng) * _atom(rng)
        if k == 1:
            return _poly(rng, 1, 3) / (X ** 2 + m)
        if k == 2:
            return a * sp.exp(b * X ** 2 + c)
        return sp.sin(b * X) * sp.cos(c * X)
    k = rng.randrange(5)
    if k == 0:
        return sp.sin(a * X ** 2 + b * X + c) * sp.exp(b * X)
    if k == 1:
        return sp.log(X ** 2 + m) / (X + c)
    if k == 2:
        return m * sp.exp(sp.sin(a * X + c))
    if k == 3:
        return sp.sqrt(a * X ** 2 + m) * sp.cos(b * X)
    return _poly(rng, 2, 3) * sp.exp(sp.cos(b * X))


def _calc(rng, level):
    mode = rng.random()
    F = _fn(rng, level)
    metadata = {"primitive": sp.sstr(F)}
    if mode < 0.4:
        return ("diff", f"Find the derivative of f(x) = {sp.sstr(F)} with respect to x.",
                sp.sstr(sp.diff(F, X)), sp.sstr(F), metadata)
    if mode < 0.75:
        f = sp.diff(F, X)
        return ("integral", f"Find an antiderivative of {sp.sstr(f)} with respect to x.",
                sp.sstr(F), sp.sstr(f), metadata)
    f = sp.diff(F, X)
    a, b = sorted(rng.sample(list(POLY_BOUNDS), 2))
    value = F.subs(X, b) - F.subs(X, a)
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not value.is_number:
        return None
    metadata.update(lower=sp.sstr(a), upper=sp.sstr(b))
    return ("defint", (f"Evaluate the definite integral of {sp.sstr(f)} with respect to x "
            f"from {sp.sstr(a)} to {sp.sstr(b)}."), sp.sstr(value), sp.sstr(f), metadata)


def _arith(rng, level):
    if level == 1:
        a, b = _mag(rng, 100, 999), _mag(rng, 100, 999)
        expr = f"{a} {rng.choice('+-')} {b}"
    elif level == 2:
        a, b = _mag(rng, 12, 99), _mag(rng, 12, 99)
        if rng.random() < 0.5:
            expr = f"{a} * {b}"
        else:
            expr = f"{a * _mag(rng, 12, 99)} / {a}"
    else:
        a, b, c, d, e = (_mag(rng, 2, 40) for _ in range(5))
        expr = rng.choice((
            f"({a} + {b}) * {c} - {d} * {e}",
            f"{a} * ({b} - {c}) + {d} * {e} - {a}",
            f"{a} ** 2 - {b} * ({c} - {d}) + {e}",
        ))
    value = sp.sympify(expr)
    if not value.is_Integer:
        return None
    return ("eval", f"Compute {expr}.", str(value), "")


def _algebra(rng, level):
    if level == 1:
        a, x0, b = _sgn(rng, 2, 9), rng.randint(-12, 12), rng.randint(-20, 20)
        c = a * x0 + b
        return ("solve", f"Solve {a}*x{_plus(b)} = {c} for x.", str(x0), "")
    if level == 2:
        if rng.random() < 0.5:
            r1, r2 = rng.sample(range(-50, 51), 2)
            a = _mag(rng, 1, 12)
            poly = sp.expand(a * (X - r1) * (X - r2))
            if rng.random() < 0.5:
                roots = sorted((r1, r2))
                return ("quadratic", f"Solve {sp.sstr(poly)} = 0 for x. List all roots in increasing order.",
                        f"[{roots[0]}, {roots[1]}]", sp.sstr(poly))
            return ("factor", f"Factor {sp.sstr(poly)} over the integers.",
                    sp.sstr(a * (X - r1) * (X - r2)), sp.sstr(poly))
        a, b, c = _sgn(rng, 1, 20), _sgn(rng, 1, 50), _sgn(rng, 1, 50)
        return ("algebra", f"Expand ({a}*x{_plus(b)})*(x{_plus(c)}).",
                sp.sstr(sp.expand((a * X + b) * (X + c))), "")
    if rng.random() < 0.5:
        p, q, s = _sgn(rng, 1, 6), _sgn(rng, 1, 6), _sgn(rng, 1, 6)
        w = _sgn(rng, 1, 6)
        if p * w - q * s == 0:
            return None
        x0, y0 = rng.randint(-9, 9), rng.randint(-9, 9)
        r, u = p * x0 + q * y0, s * x0 + w * y0
        q1 = f"{p}*x {'+' if q >= 0 else '-'} {abs(q)}*y = {r}"
        q2 = f"{s}*x {'+' if w >= 0 else '-'} {abs(w)}*y = {u}"
        return ("solve", f"Solve the system: {q1}, {q2} for x and y.", f"x = {x0}, y = {y0}", "")
    r, a, b = _sgn(rng, 2, 50), _mag(rng, 1, 12), rng.randint(-30, 30)
    numerator = sp.expand((X - r) * (a * X + b))
    return ("simplify", f"Simplify ({sp.sstr(numerator)})/(x - ({r})) for x != {r}.",
            sp.sstr(a * X + b), "")


def _precalc(rng, level):
    if level == 1:
        name = rng.choice(list(TRIG))
        theta = rng.choice(ANGLES)
        a, b = _sgn(rng, 1, 99), rng.randint(-99, 99)
        turns = rng.randint(-5, 5)
        angle = theta + 2 * turns * sp.pi
        value = a * TRIG[name](theta) + b
        if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return None
        return ("trig", f"Compute {a}*{name}({sp.sstr(angle)}) + ({b}).", sp.sstr(value), "")
    if level == 2:
        k = rng.randrange(4)
        b = rng.choice((2, 3, 5, 7, 10, 11))
        a, c = _sgn(rng, 1, 30), rng.randint(-50, 50)
        if k == 0:
            p, q = _mag(rng, 2, 8), _mag(rng, 2, 8)
            return ("log", f"Evaluate {a}*log_{b}({b ** p} * {b ** q}) + ({c}).", str(a * (p + q) + c), "")
        if k == 1:
            p = _mag(rng, 2, 8)
            return ("log", f"Evaluate {a} times log base {b} of {b ** p}, plus ({c}).", str(a * p + c), "")
        if k == 2:
            u = b * X + rng.randint(1, 30)
            expr = a * sp.sin(u) ** 2 + a * sp.cos(u) ** 2 + c
            return ("identity", f"Simplify {sp.sstr(expr)}.", str(a + c), sp.sstr(expr))
        u = b * X + rng.randint(1, 30)
        expr = a * sp.tan(u) * sp.cos(u) + c
        return ("identity", f"Simplify {sp.sstr(expr)}.", sp.sstr(a * sp.sin(u) + c), sp.sstr(expr))
    k = rng.randrange(3)
    a, c = _mag(rng, 1, 30), rng.randint(-30, 30)
    if k == 0:
        b, kk = rng.choice((2, 3, 5, 7)), _mag(rng, 1, 12)
        return ("solve", f"Solve {b}**({a}*x + ({c})) = {b ** kk} for real x.",
                sp.sstr(sp.Rational(kk - c, a)), "")
    if k == 1:
        b, kk = rng.choice((2, 3, 5, 7)), _mag(rng, 1, 8)
        return ("solve", f"Solve log base {b} of ({a}*x + ({c})) = {kk} for real x.",
                sp.sstr(sp.Rational(b ** kk - c, a)), "")
    v, theta = rng.choice(list(TRIG_SOLUTIONS.items()))
    answer = (sp.sympify(theta) - c) / a
    return ("solve", f"Solve sin({a}*x + ({c})) = {v} where 0 <= {a}*x + ({c}) <= pi/2.",
            sp.sstr(answer), "")


GENERATORS = {"arithmetic": _arith, "algebra": _algebra, "precalculus": _precalc, "calculus": _calc}


def generate(topic, level, n, seed=0, *, excluded_sources=()):
    if topic not in TOPICS or level not in LEVELS:
        raise ValueError(f"expected a topic in {TOPICS} and a level in {LEVELS}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    rng = random.Random(f"{seed}:{topic}:{level}")
    gen = GENERATORS[topic]
    seen, items, attempts = set(excluded_sources), [], 0
    while len(items) < n:
        attempts += 1
        if attempts > 200 * n:
            raise RuntimeError(f"could not generate {n} unique items for {topic} level {level}")
        try:
            rec = gen(rng, level)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if rec is None:
            continue
        kind, question, answer, expr = rec[:4]
        metadata = rec[4] if len(rec) > 4 else {}
        source = metadata.get("primitive") if topic == "calculus" else question
        digest = hashlib.sha256(f"{topic}|{source}".encode()).hexdigest()[:20]
        sid = f"{topic}-{digest}"
        if sid in seen:
            continue
        seen.add(sid)
        items.append(MathItem(f"{topic}-L{level}-{digest}", topic, level, kind, question, answer, expr,
                              source_id=sid, **metadata))
    return items


def load_hendrycks_precalc(split="train"):
    from datasets import load_dataset

    items = []
    for i, row in enumerate(load_dataset("EleutherAI/hendrycks_math", "precalculus", split=split)):
        answer = last_boxed(row["solution"])
        digits = re.search(r"\d+", str(row["level"]))
        if not answer or not digits or int(digits.group()) not in HENDRYCKS_LEVEL:
            continue
        items.append(MathItem(f"hendrycks-precalc-{split}-{i}", "precalculus",
                              HENDRYCKS_LEVEL[int(digits.group())], "latex", row["problem"], answer))
    return items


def build_math(root, seed=0, sizes=None, *, include_hendrycks=False, topics=TOPICS, levels=LEVELS):
    root = ensure_writable(root)
    sizes = SIZES if sizes is None else sizes
    if set(sizes) != {"train", "dev", "test"}:
        raise ValueError("sizes must contain train, dev, and test")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in sizes.values()):
        raise ValueError("sizes must be nonnegative integers")
    total = sum(sizes.values())
    counts, used = {}, set()
    for topic in topics:
        for level in levels:
            rows = [it.to_alpaca() for it in generate(topic, level, total, seed, excluded_sources=used)]
            used.update(source_id(row) for row in rows)
            parts = split_counts(rows, sizes, seed)
            assert_disjoint(parts)
            for name, part in parts.items():
                part = [dict(row, split=name) for row in part]
                write_jsonl(root / topic / f"level{level}" / f"{name}.jsonl", part)
                counts[f"{topic}/level{level}/{name}"] = len(part)
    if include_hendrycks:
        for original_split in ("train", "test"):
            rows = [it.to_alpaca() for it in load_hendrycks_precalc(original_split)]
            for row in rows:
                row["source_id"] = "hendrycks-" + hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:20]
            parts = group_split(rows, {"train": 0.8, "dev": 0.2}, seed=seed) if original_split == "train" else {"test": rows}
            for name, part in parts.items():
                write_jsonl(root / "precalculus" / "hendrycks" / f"{name}.jsonl",
                            (dict(row, split=name, dataset="EleutherAI/hendrycks_math") for row in part))
                counts[f"precalculus/hendrycks/{name}"] = len(part)
    return counts
