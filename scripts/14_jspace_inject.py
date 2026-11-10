"""PLAN attack 14: J-space concept injection on calculus ("concept-hole" prompts).

Each calculus problem is rewritten so the operation word is replaced by a placeholder ("Apply operation Z to
f(x) = ... with respect to x [on the interval [a, b]]."; no calculus words). The J-space direction of the operation
concept (``v = J_l^T u_t``, mean of the unit pullbacks of ' derivative'/' differentiate' or ' integral'/' integrate')
is added to the residual at the placeholder token (pos=hole) or at every prompt position after BOS (pos=all), at one
layer or a band, as ``alpha * ||h_p|| * v_hat`` (src/lfm3u/jspace/inject.py inject_rows). Only the prompt prefill is
edited; generated tokens are not. Greedy, thinking off, chat template, no system prompt.

Stages
  sweep  (target only, DEV split math_calc_dev.jsonl, n per kind): controls a/b plus the grid
         pos x layer-spec x alpha with the correct concept; picks the config with the best mean-over-kinds accuracy
         (ties: lower gibberish, fewer layers, smaller alpha) -> <out>/config.json
  eval   100 TEST problems (34 diff, 33 integral, 33 defint) under the conditions
           a_explicit   original prompt, no injection
           b_hole       placeholder prompt, no injection
           inj          placeholder + correct concept at the chosen config
           c_random     placeholder + random unit direction (same alpha / layers / positions)
           d_wrong      placeholder + the wrong concept (integral on diff problems, derivative on integral/defint)
         -> <out>/summary.json, <out>/generations/<cond>.jsonl

Every row also records which operation the answer matches: op_diff (answer == f'(x)) and op_int (answer is an
antiderivative of f; for defint the definite value), and whether the output names a derivative / integral.

  python scripts/14_jspace_inject.py --model PATH --name target --lens lens.pt --out results/jspace/inject_calc/target
         --stages sweep,eval            # target
  python scripts/14_jspace_inject.py ... --stages eval --config results/jspace/inject_calc/target/config.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm3u import grade  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data import math_gen as mg  # noqa: E402

CONCEPTS = {"diff": (" derivative", " differentiate"), "int": (" integral", " integrate")}
FORBIDDEN = re.compile(r"deriv|differentia|integra|antideriv|d/dx|\bdx\b", re.I)
MARKER_CTX, MARKER = "operation ", "Z"
TAIL = " Give the final answer on the last line as 'Answer: <expression>'."
STRIP = ("<|im_end|>", "<|endoftext|>", "<|startoftext|>", "<|im_start|>")
NAMES_DIFF = re.compile(r"derivative|differentiat", re.I)  # run inj_1 used r"deriv|differentia" (also hits "derive")
NAMES_INT = re.compile(r"integra|antideriv", re.I)
TEST_COUNTS = {"diff": 34, "integral": 33, "defint": 33}


def seeded(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def read_jsonl(p) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick(rows: list[dict], counts: dict[str, int], tag: str) -> list[dict]:
    out = []
    for kind, n in counts.items():
        rs = [r for r in rows if r["kind"] == kind]
        seeded("inj14", tag, kind).shuffle(rs)
        out += rs[:n]
    return out


def hole_prompt(r: dict) -> str:
    rng = f" on the interval [{r['lower']}, {r['upper']}]" if r["kind"] == "defint" else ""
    p = f"Apply {MARKER_CTX}{MARKER} to f(x) = {r['expr']} with respect to x{rng}.{TAIL}"
    if FORBIDDEN.search(p):
        raise ValueError(f"calculus word in placeholder prompt: {p}")
    return p


def right_op(kind: str) -> str:
    return "diff" if kind == "diff" else "int"


def wrong_op(kind: str) -> str:
    return "int" if kind == "diff" else "diff"


def parse_specs(s: str) -> list[list[int]]:
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = map(int, item.split("-"))
            out.append(list(range(a, b + 1)))
        else:
            out.append([int(item)])
    return out


def spec_name(layers: list[int]) -> str:
    return str(layers[0]) if len(layers) == 1 else f"{layers[0]}-{layers[-1]}"


# ------------------------------------------------------------------ grading

_DERIV: dict[str, str | None] = {}


def derivative_of(expr: str) -> str | None:
    if expr not in _DERIV:
        try:
            e = mg.parse_expression(expr)
            x = next((s for s in e.free_symbols if s.name == "x"), sp.Symbol("x"))
            _DERIV[expr] = sp.sstr(sp.diff(e, x))
        except Exception:  # noqa: BLE001
            _DERIV[expr] = None
    return _DERIV[expr]


def grade_row(r: dict, text: str) -> dict:
    g = grade_item(r, text, thinking=False)
    d = derivative_of(r["expr"])
    if r["kind"] == "diff":
        op_diff = g["correct"]
        op_int = grade_item({"kind": "integral", "expr": r["expr"]}, text)["correct"]
    else:
        op_int = g["correct"]
        op_diff = bool(d) and grade_item({"kind": "diff", "answer": d}, text)["correct"]
    return {"correct": g["correct"], "gibberish": g["gibberish"], "answer_text": g["answer_text"], "parsed": g["parsed"],
            "op_diff": bool(op_diff), "op_int": bool(op_int),
            "names_diff": bool(NAMES_DIFF.search(text)), "names_int": bool(NAMES_INT.search(text))}


def summarize(recs: list[dict]) -> dict:
    def m(key, rs):
        return round(sum(bool(r[key]) for r in rs) / max(1, len(rs)), 3)

    by_kind = defaultdict(list)
    for r in recs:
        by_kind[r["kind"]].append(r)
    keys = ("correct", "gibberish", "truncated", "op_diff", "op_int", "names_diff", "names_int")
    out = {"n": len(recs), **{("acc" if k == "correct" else k): m(k, recs) for k in keys}}
    out["by_kind"] = {k: {"n": len(rs), "acc": m("correct", rs), "gibberish": m("gibberish", rs),
                          "truncated": m("truncated", rs), "op_diff": m("op_diff", rs), "op_int": m("op_int", rs)}
                      for k, rs in sorted(by_kind.items())}
    out["acc_mean_over_kinds"] = round(sum(v["acc"] for v in out["by_kind"].values()) / max(1, len(by_kind)), 3)
    return out


# ------------------------------------------------------------------ model side

class Runner:
    def __init__(self, a):
        from jlens import JacobianLens

        from lfm3u.jspace.fit import load_model, wrap
        from lfm3u.jspace.readout import lens_to

        self.a = a
        self.hf, self.tok = load_model(a.model)
        self.jm = wrap(self.hf, self.tok)
        self.device = next(self.hf.parameters()).device
        self.lens = JacobianLens.load(a.lens)
        lens_to(self.lens, self.device)
        self.lens_layers = sorted(self.lens.jacobians)
        self.d = self.hf.config.hidden_size
        self.concept_ids = {}
        for op, words in CONCEPTS.items():
            ids = []
            for w in words:
                t = self.tok(w, add_special_tokens=False).input_ids
                if len(t) != 1:
                    raise ValueError(f"concept word {w!r} is not one token: {t}")
                ids.append(t[0])
            self.concept_ids[op] = ids
        self._dirs: dict = {}
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos = sorted(i for i in {self.tok.eos_token_id, im_end} if isinstance(i, int) and i >= 0)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.strip = tuple(s for s in (*STRIP, self.tok.eos_token, self.tok.pad_token) if s)

    def direction(self, op: str, layer: int) -> torch.Tensor:
        from lfm3u.jspace.inject import concept_direction, random_direction

        key = (op, layer)
        if key not in self._dirs:
            v = (random_direction(self.d, seed=1000 + layer) if op == "rand"
                 else concept_direction(self.lens, self.jm, self.concept_ids[op], layer))
            self._dirs[key] = v.to(self.device, torch.float32)
        return self._dirs[key]

    def encode(self, prompt: str, pos: str | None) -> tuple[list[int], list[bool]]:
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            tokenize=False, enable_thinking=False)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if pos == "all":
            mask = [i > 0 for i in range(len(ids))]
        elif pos == "hole":
            s = text.index(MARKER_CTX + MARKER) + len(MARKER_CTX)
            e = s + len(MARKER)
            mask = [bool(a < e and b > s) for a, b in offs]
            if not any(mask):
                raise ValueError("placeholder token not found")
        else:
            mask = [False] * len(ids)
        return ids, mask

    @torch.no_grad()
    def generate(self, items: list[dict], max_tokens: int, bs: int) -> None:
        """Fills item['text'], item['n_tokens'], item['truncated'] in place."""
        from lfm3u.jspace.inject import inject_rows

        order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
        t0 = time.time()
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            B, L = len(idx), max(len(items[i]["ids"]) for i in idx)
            inp = torch.full((B, L), self.pad, dtype=torch.long)
            att = torch.zeros((B, L), dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.bool)
            alphas = torch.zeros(B)
            layers = sorted({l for i in idx if items[i]["edit"] for l in items[i]["edit"]["layers"]})
            dirs = {l: torch.zeros(B, self.d, device=self.device) for l in layers}
            for j, i in enumerate(idx):
                it = items[i]
                n = len(it["ids"])
                inp[j, L - n:] = torch.tensor(it["ids"])
                att[j, L - n:] = 1
                mask[j, L - n:] = torch.tensor(it["mask"])
                if it["edit"]:
                    alphas[j] = it["edit"]["alpha"]
                    for l in it["edit"]["layers"]:
                        dirs[l][j] = self.direction(it["edit"]["op"], l)
            gk = dict(max_new_tokens=max_tokens, pad_token_id=self.pad, eos_token_id=self.eos, do_sample=False,
                      temperature=None, top_p=None, top_k=None)
            with inject_rows(self.hf.model.layers, dirs, alphas.to(self.device), mask.to(self.device)):
                g = self.hf.generate(input_ids=inp.to(self.device), attention_mask=att.to(self.device), **gk)
            for j, i in enumerate(idx):
                new = g[j, L:].tolist()
                n = next((k for k, t in enumerate(new) if t in self.eos or t == self.pad), len(new))
                text = self.tok.decode(new[:n], skip_special_tokens=False)
                for s in self.strip:
                    text = text.replace(s, "")
                items[i].update(text=text.strip(), n_tokens=n, truncated=n >= max_tokens)
            print(f"[inj14] batch {b // bs + 1}/{(len(order) + bs - 1) // bs}: {B} rows, L={L}, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)

    def item(self, r: dict, cond: str, edit: dict | None, cfg: str = "") -> dict:
        prompt = r["instruction"] if cond == "a_explicit" else hole_prompt(r)
        pos = edit["pos"] if edit else None
        ids, mask = self.encode(prompt, pos)
        return {"row": r, "cond": cond, "cfg": cfg, "edit": edit, "prompt": prompt, "ids": ids, "mask": mask}


def finish(items: list[dict]) -> list[dict]:
    recs = []
    for it in items:
        r = it["row"]
        recs.append({"id": r["id"], "kind": r["kind"], "level": r["level"], "cond": it["cond"], "cfg": it["cfg"],
                     "edit": it["edit"], "prompt": it["prompt"], "n_hole_tokens": sum(it["mask"]),
                     "output": it["text"], "n_tokens": it["n_tokens"], "truncated": it["truncated"],
                     "gold": r["answer"], **grade_row(r, it["text"])})
    return recs


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cfg_key(edit: dict) -> str:
    return f"{edit['pos']}|L{spec_name(edit['layers'])}|a{edit['alpha']}"


def run_sweep(R: Runner, a, out: Path) -> dict:
    dev = pick(read_jsonl(a.dev), {"diff": a.dev_per_kind, "integral": a.dev_per_kind, "defint": a.dev_per_kind}, "dev")
    specs = [s for s in parse_specs(a.layers) if all(l in R.lens_layers for l in s)]
    if not specs:
        raise ValueError(f"no layer spec inside the lens layers {R.lens_layers}")
    alphas = [float(x) for x in a.alphas.split(",")]
    items = [R.item(r, c, None, c) for r in dev for c in ("a_explicit", "b_hole")]
    for pos in a.positions.split(","):
        for spec in specs:
            for al in alphas:
                for r in dev:
                    e = {"pos": pos, "layers": spec, "alpha": al, "op": right_op(r["kind"])}
                    items.append(R.item(r, "inj", e, cfg_key(e)))
    print(f"[inj14] sweep: {len(dev)} dev problems, {len(items)} generations", flush=True)
    R.generate(items, a.sweep_max_tokens, a.batch)
    recs = finish(items)
    write_jsonl(out / "generations" / "sweep.jsonl", recs)
    groups = defaultdict(list)
    for r in recs:
        groups[r["cfg"]].append(r)
    table = {k: summarize(v) for k, v in groups.items()}
    cands = []
    for k, v in table.items():
        if k in ("a_explicit", "b_hole"):
            continue
        e = groups[k][0]["edit"]
        cands.append((-v["acc_mean_over_kinds"], v["gibberish"], len(e["layers"]), e["alpha"], k, e))
    cands.sort(key=lambda t: t[:5])
    best = cands[0]
    chosen = {k: best[5][k] for k in ("pos", "layers", "alpha")}
    res = {"name": a.name, "model": a.model, "lens": a.lens, "split": str(a.dev), "n_dev": len(dev),
           "dev_ids": [r["id"] for r in dev], "max_tokens": a.sweep_max_tokens, "selection":
           "max acc_mean_over_kinds; ties: lower gibberish, fewer layers, smaller alpha", "chosen": chosen,
           "chosen_key": best[4], "chosen_dev": table[best[4]], "controls_dev": {k: table[k] for k in ("a_explicit", "b_hole")},
           "ranking": [{"cfg": c[4], "acc_mean_over_kinds": -c[0], "gibberish": c[1]} for c in cands], "table": table}
    (out / "sweep.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"[inj14] chosen {best[4]}: dev acc {table[best[4]]['acc_mean_over_kinds']} "
          f"(b_hole {table['b_hole']['acc_mean_over_kinds']}, a_explicit {table['a_explicit']['acc_mean_over_kinds']})",
          flush=True)
    return chosen


def run_eval(R: Runner, a, out: Path, cfg: dict) -> dict:
    counts = TEST_COUNTS if not a.n_test_per_kind else {k: a.n_test_per_kind for k in TEST_COUNTS}
    test = pick(read_jsonl(a.test), counts, "test")
    bad = [l for l in cfg["layers"] if l not in R.lens_layers]
    if bad:
        raise ValueError(f"config layers {bad} not in lens")
    items = []
    for r in test:
        items.append(R.item(r, "a_explicit", None, "a_explicit"))
        items.append(R.item(r, "b_hole", None, "b_hole"))
        for cond, op in (("inj", right_op(r["kind"])), ("c_random", "rand"), ("d_wrong", wrong_op(r["kind"]))):
            items.append(R.item(r, cond, {**cfg, "op": op}, cond))
    print(f"[inj14] eval: {len(test)} test problems, {len(items)} generations, config {cfg}", flush=True)
    t0 = time.time()
    R.generate(items, a.max_tokens, a.batch)
    recs = finish(items)
    by = defaultdict(list)
    for r in recs:
        by[r["cond"]].append(r)
    for c, rs in by.items():
        write_jsonl(out / "generations" / f"{c}.jsonl", rs)
    res = {"name": a.name, "model": a.model, "lens": a.lens, "config": cfg, "concept_tokens": {k: list(v) for k, v in CONCEPTS.items()},
           "concept_ids": R.concept_ids, "n_test": len(test), "test_counts": counts, "max_tokens": a.max_tokens,
           "greedy": True, "thinking": False, "generate_seconds": round(time.time() - t0, 1),
           "hole_template": f"Apply {MARKER_CTX}{MARKER} to f(x) = <expr> with respect to x[ on the interval [a, b]].{TAIL}",
           "conditions": {c: summarize(by[c]) for c in ("a_explicit", "b_hole", "inj", "c_random", "d_wrong")}}
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for c, s in res["conditions"].items():
        print(f"[inj14] {a.name} {c:10s} acc {s['acc']:.3f} gib {s['gibberish']:.3f} trunc {s['truncated']:.3f} "
              f"op_diff {s['op_diff']:.3f} op_int {s['op_int']:.3f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="eval", help="sweep,eval | eval")
    ap.add_argument("--config", default=None, help="eval only: config.json written by the target's sweep")
    ap.add_argument("--dev", default="attack_data/_inputs/math_calc_dev.jsonl")
    ap.add_argument("--test", default="attack_data/_inputs/math_calc_test.jsonl")
    ap.add_argument("--dev-per-kind", type=int, default=6)
    ap.add_argument("--n-test-per-kind", type=int, default=0, help="0 = 34/33/33")
    ap.add_argument("--layers", default="5,6,7,8,9,10,11,12,13,14,15,16,5-8,9-12,13-16,5-12")
    ap.add_argument("--alphas", default="0.25,0.5,1,2")
    ap.add_argument("--positions", default="hole,all")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sweep-max-tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = a.stages.split(",")
    R = Runner(a)
    cfg = run_sweep(R, a, out) if "sweep" in stages else None
    if "eval" in stages:
        if cfg is None:
            if not a.config:
                sys.exit("--stages eval needs --config (or run sweep first)")
            cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        run_eval(R, a, out, cfg)


if __name__ == "__main__":
    main()
