"""Phase 7 driver: agentic weight surgery (src/lfm3u/surgery). State persists in results/surgery/<run>/.

  new RUN --target book|calculus --budget 10|25|50 [--note TEXT]   create a run (forward budget is fixed here)
  exec RUN STEP.py        run a step file with `S` (the Session) in scope; the file and its output are copied to
                          results/surgery/RUN/steps/NN_<name>.{py,out} as the agent's trace
  status RUN              budgets used and active edits
  evalckpt DIR --out OUT [--tasks full] [--cfg JSON]    dev eval of a saved checkpoint (not budgeted; the audit side)
  probe DIR --out OUT [--target calculus|book] [--steps 10,25,50] [--kind mixed|diff|integral|defint] [--cfg JSON]
                          relearning probe (lfm3u.surgery.relearn) on a saved checkpoint: not budgeted, the audit side;
                          writes OUT/probe_<kind>_s<seed>.json and the generations per step
  cert RUN [--uri s3://...]                             write results/surgery/RUN/certificate.md
  selftest [--model]      numeric checks of the edit math and the guard (--model adds the checks on the real model,
                          including that the relearning probe leaves every weight bit-identical)

Step files are plain Python. In scope: S, torch, score (lfm3u.surgery.score), show(obj) (prints JSON), ROOT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def show(obj, **kw) -> None:
    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)
    print(json.dumps(obj, indent=kw.get("indent"), default=default, ensure_ascii=False), flush=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def cmd_exec(a) -> None:
    import torch

    from lfm3u.surgery import score
    from lfm3u.surgery.tools import Session

    step = Path(a.step)
    code = step.read_text(encoding="utf-8")
    steps = ROOT / "results/surgery" / a.run / "steps"
    if not (steps.parent / "state.json").is_file():
        sys.exit(f"no run {a.run}; create it with `new`")
    steps.mkdir(exist_ok=True)
    n = len(list(steps.glob("*.py"))) + 1
    stem = f"{n:02d}_{step.stem}"
    (steps / f"{stem}.py").write_text(code, encoding="utf-8", newline="\n")
    with (steps / f"{stem}.out").open("w", encoding="utf-8", newline="\n") as f:
        old = sys.stdout
        sys.stdout = Tee(old, f)
        try:
            t0 = time.time()
            S = Session(a.run)
            g = S.guard
            print(f"[step {stem}] loaded in {time.time() - t0:.0f}s; run {a.run} ({g.state['target']}), calls {g.state['calls_used']}/"
                  f"{g.state['call_budget']}, {len(g.active())} active edits, edit cost {g.used()}/{g.state['edit_budget']}", flush=True)
            exec(compile(code, str(step), "exec"), {"S": S, "torch": torch, "score": score, "show": show, "ROOT": ROOT, "__name__": "__step__"})
            print(f"[step {stem}] done in {time.time() - t0:.0f}s; calls {S.guard.state['calls_used']}/{S.guard.state['call_budget']}, "
                  f"edit cost {S.guard.used()}/{S.guard.state['edit_budget']}", flush=True)
        finally:
            sys.stdout = old


def cmd_evalckpt(a) -> None:
    from lfm3u.surgery import score
    from lfm3u.surgery.tools import load_model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(a.dir)
    res = score.evaluate(model, tok, a.tasks if a.tasks in score.PRESETS else a.tasks.split(","), json.loads(a.cfg) if a.cfg else None, out)
    res |= {"model": str(a.dir), "tasks_arg": a.tasks}
    (out / "dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    show(res["brief"])


def cmd_probe(a) -> None:
    from lfm3u.surgery import relearn
    from lfm3u.surgery.tools import load_model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(a.dir)
    tag = f"probe_{a.kind}_s{a.seed}"
    res = relearn.probe(model, tok, target=a.target, steps=[int(s) for s in a.steps.split(",")], n_items=a.n_items, kind=a.kind,
                        seed=a.seed, cfg=json.loads(a.cfg) if a.cfg else None, out_dir=out / tag)
    res |= {"model": str(a.dir), "charged": False}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    show(res["brief"], indent=1)


def selftest(with_model: bool) -> None:
    import shutil
    import tempfile

    import torch

    from lfm3u.surgery import edits as E
    from lfm3u.surgery.guard import BudgetExceeded, Guard, GuardError, edit_cost

    torch.manual_seed(0)
    dd = torch.float64
    # 1. writer side: under g * y / rms(y), projecting off u = unit(g * r) zeroes the write along r; projecting off r does not
    rows, cols = 64, 200
    W = torch.randn(rows, cols, dtype=dd)
    g = torch.rand(rows, dtype=dd) * 0.1 + 1e-5
    r = E.unit(torch.randn(rows, dtype=dd))
    A = torch.randn(500, cols, dtype=dd)

    def write_along_r(Wx):
        y = A @ Wx.T
        z = g * y / y.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
        return float(((z @ r).abs() / z.norm(dim=-1)).max())

    U, V = E.ablate_out(W, r, g)
    Un, Vn = E.ablate_out(W, r, None)
    good, naive, before = write_along_r(W + U @ V.T), write_along_r(W + Un @ Vn.T), write_along_r(W)
    assert good < 1e-12, good
    assert naive > 1e-3, naive
    print(f"[selftest] writer: |z.r|/|z| max  unedited {before:.3g}  naive (I - rr^T)W {naive:.3g}  sandwich (I - uu^T)W {good:.3g}")
    # 2. reader side: W'(g_in * x) = W(g_in * (x - r r^T x))
    Wr = torch.randn(300, rows, dtype=dd)
    gi = torch.rand(rows, dtype=dd) * 0.3 + 0.02
    X = torch.randn(500, rows, dtype=dd)
    U, V = E.ablate_in(Wr, r, gi)
    lhs = (gi * X) @ (Wr + U @ V.T).T
    rhs = (gi * (X - (X @ r)[:, None] * r)) @ Wr.T
    err = float((lhs - rhs).abs().max() / rhs.abs().max())
    assert err < 1e-10, err
    print(f"[selftest] reader: max relative error {err:.3g}")
    # 3. guard
    assert edit_cost("lowrank", (2048, 7680), rank=1) == 9728 and edit_cost("cols", (2048, 7680), n_cols=3) == 6144
    tmp = Path(tempfile.mkdtemp(prefix="guardtest_", dir=str(ROOT / "results")))
    try:
        try:
            Guard(tmp / "r", n_params=1_000_000, call_budget=7)
            raise AssertionError("budget 7 accepted")
        except GuardError:
            pass
        gd = Guard(tmp / "r", n_params=1_000_000, call_budget=10, target="book")
        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.embedding_norm.weight"):
            try:
                gd.admit(param=name, shape=(4, 4), kind="lowrank", cost=8, op="rank_one", args={}, rationale="should be rejected")
                raise AssertionError(name)
            except GuardError:
                pass
        gd.admit(param="model.layers.0.feed_forward.w2.weight", shape=(10, 50), kind="lowrank", cost=600, op="rank_one", args={}, rationale="within budget")
        try:
            gd.admit(param="model.layers.1.feed_forward.w2.weight", shape=(10, 50), kind="lowrank", cost=600, op="rank_one", args={}, rationale="over budget")
            raise AssertionError("edit budget not enforced")
        except BudgetExceeded:
            pass
        gd.revert("e001", "test")
        gd.admit(param="model.layers.1.feed_forward.w2.weight", shape=(10, 50), kind="lowrank", cost=600, op="rank_one", args={}, rationale="fits after revert")
        try:
            gd.charge("run_forward", 257)
            raise AssertionError("prompt cap")
        except GuardError:
            pass
        for _ in range(6):
            gd.charge("run_forward", 10)
        gd2 = Guard(tmp / "r")  # a new process sees the same counter
        assert gd2.state["calls_used"] == 6 and gd2.used() == 600
        # relearning probe: charged several calls at once, refused as a whole when they do not fit, never cheaper than 3
        try:
            gd2.charge("relearn_probe", 20, cost=5)
            raise AssertionError("probe over budget accepted")
        except BudgetExceeded:
            pass
        try:
            gd2.charge("relearn_probe", 20, cost=1)
            raise AssertionError("probe charged one call")
        except GuardError:
            pass
        assert gd2.state["calls_used"] == 6
        assert gd2.charge("relearn_probe", 20, cost=3) == 9 and gd2.state["calls"][-1]["cost"] == 3
        gd2.record_probe({"kind": "mixed", "brief": {}})
        assert Guard(tmp / "r").state["probes"][0]["call"] == 9
        gd2.charge("dev_score", 0)
        try:
            Guard(tmp / "r").charge("dev_score", 0)
            raise AssertionError("forward budget not enforced")
        except BudgetExceeded:
            pass
        try:
            Guard(tmp / "r", call_budget=25)
            raise AssertionError("budget changed")
        except GuardError:
            pass
        gd2.close({"dir": "x"})
        try:
            Guard(tmp / "r").admit(param="model.layers.2.feed_forward.w2.weight", shape=(10, 50), kind="lowrank", cost=1, op="rank_one", args={}, rationale="closed run")
            raise AssertionError("closed run accepted an edit")
        except GuardError:
            pass
        print("[selftest] guard: protected tensors, edit budget, revert, prompt cap, persisted forward counter, probe cost, fixed budget, close: ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not with_model:
        return
    # 4. real model: Lfm3RMSNorm module + real weights; bf16 rounding; persistence; revert
    from lfm3u.surgery import score
    from lfm3u.surgery.tools import Session

    run = "_selftest"
    shutil.rmtree(ROOT / "results/surgery" / run, ignore_errors=True)
    S = Session(run, target="calculus", budget=10, note="selftest")
    try:
        tok = S.tok
        fp = [score.chat_ids(tok, f"Find the derivative of f(x) = {e}.") for e in ("x^3 + 2*x", "sin(3*x)*exp(x)", "log(x^2 + 1)", "x*cos(x)")]
        cp = [score.chat_ids(tok, q) for q in ("Compute 17 * 23.", "Solve 3*x + 5 = 20 for x.", "Simplify (x^2 - 1)/(x - 1).", "What is 144 / 12?")]
        r = S.direction(fp, cp, 4, name="r4")["r"]
        layer = S.model.model.layers[4]
        name, gname = "model.layers.4.feed_forward.w2.weight", "model.layers.4.post_ffn_norm.weight"
        W32, g32 = S.effective(name), S.effective(gname)
        hid = S.activations(fp + cp, ["in:model.layers.4.feed_forward.w2"], cov=True)
        a = torch.randn(256, W32.shape[1], device=S.device) * hid["in:model.layers.4.feed_forward.w2#m2"].diagonal().sqrt().to(S.device)
        norm32 = type(layer.post_ffn_norm)(W32.shape[0], eps=layer.post_ffn_norm.variance_epsilon).to(S.device).float()
        norm32.weight.data.copy_(g32)
        rr = r.to(S.device)

        def rel(Wx):
            z = norm32(a @ Wx.T)
            return float(((z @ rr).abs() / z.norm(dim=-1)).mean())

        U, V = E.ablate_out(W32, rr, g32)
        Un, Vn = E.ablate_out(W32, rr, None)
        b0, b_naive, b_good = rel(W32), rel(W32 + Un @ Vn.T), rel(W32 + U @ V.T)
        b_bf16 = rel((W32 + U @ V.T).to(torch.bfloat16).float())
        print(f"[selftest] layer-4 w2 with Lfm3RMSNorm, mean |z.r|/|z|: unedited {b0:.3g}  naive {b_naive:.3g}  sandwich fp32 {b_good:.3g}  sandwich bf16 {b_bf16:.3g}")
        assert b_good < 1e-4 * b0 and b_bf16 < 0.02 * b0 and b_naive > 10 * b_bf16
        # in the running bf16 model: hook the post-norm output on real prompts
        cap = {}
        h = layer.post_ffn_norm.register_forward_hook(lambda _m, _i, o: cap.__setitem__("z", o))

        def live():
            ids = torch.tensor([fp[1]], device=S.device)
            with torch.no_grad():
                S.model.model(input_ids=ids)
            z = cap["z"][0].float()
            return float(((z @ rr).abs() / z.norm(dim=-1)).mean())

        before = live()
        orig = S.params[name].detach().clone()
        rec = S.ablate_direction(name, "r4", rationale="selftest: remove r4 from the layer-4 MLP write")
        after = live()
        S2 = Session(run)  # a second load re-applies the delta from disk
        same = torch.equal(S2.params[name], S.params[name])
        del S2
        if S.device.type == "cuda":
            torch.cuda.empty_cache()
        # relearning probe (P5), run while the edit is active: charged 3 calls, and every weight (edited ones included)
        # keeps its exact bits. Full element-wise comparison against a CPU copy, not the probe's own checksum.
        snap = {n: p.detach().cpu().clone() for n, p in S.model.named_parameters()}
        used = S.guard.state["calls_used"]
        pr = S.relearn_probe(steps=(0, 2), n_items=4, max_len=384, cfg={"calc_per_level": 1, "calc_max_new": 48}, note="selftest")
        now = dict(S.model.named_parameters())
        identical = list(now) == list(snap) and all(torch.equal(p.detach().cpu(), snap[n]) for n, p in now.items())
        same_objects = all(now[n] is p for n, p in S.params.items())
        no_adapter = not any("lora" in n.lower() for n, _ in S.model.named_modules())
        charged = S.guard.state["calls_used"] - used
        del snap
        print(f"[selftest] relearning probe (LoRA, 2 steps, edit active): charged {charged} calls; weights bit-identical: {identical}; "
              f"same parameter objects: {same_objects}; adapter removed: {no_adapter}; scored steps {sorted(pr)}; "
              f"train loss {pr['2']['train_loss']}")
        assert identical and same_objects and no_adapter and charged == 3 and sorted(pr) == ["0", "2"]
        # sparse edit of a per-layer norm gain (P5) + the gain-adapter probe: the zeroed entries stay zero, every weight
        # keeps its bits, the revert restores the gain, and the final norm stays protected
        gn = "model.layers.4.post_ffn_norm.weight"
        g0 = S.params[gn].detach().clone()
        rec2 = S.set_elements(gn, [3, 17], 0.0, rationale="selftest: close two stream coordinates of the layer-4 MLP write")
        zeroed = bool((S.params[gn][[3, 17]] == 0).all()) and int((S.params[gn] != g0).sum()) <= 2 and rec2["cost"] == 2
        snap = {n: p.detach().cpu().clone() for n, p in S.model.named_parameters()}
        pr2 = S.relearn_probe(steps=(1,), n_items=4, max_len=384, gains=True, cfg={"calc_per_level": 1, "calc_max_new": 48}, note="selftest")
        now = dict(S.model.named_parameters())
        identical2 = list(now) == list(snap) and all(torch.equal(p.detach().cpu(), snap[n]) for n, p in now.items())
        del snap
        moved = S.guard.state["probes"][-1]["gains"]
        S.revert(rec2["id"], "selftest")
        back = torch.equal(S.params[gn], g0)
        try:
            S.set_elements("model.embedding_norm.weight", [0], 0.0, rationale="must be rejected by the guard")
            raise AssertionError("final-norm edit accepted")
        except GuardError:
            pass
        print(f"[selftest] norm-gain edit: entries zeroed, cost 2: {zeroed}; probe with gain adapters ({moved['n_vectors']} vectors, "
              f"max |delta| {moved['max_abs_delta']}) leaves weights bit-identical: {identical2}; revert restores the gain: {back}; "
              f"final norm protected; scored steps {sorted(pr2)}")
        assert zeroed and identical2 and back and moved["max_abs_delta"] > 0
        S.revert(rec["id"], "selftest")
        restored = torch.equal(S.params[name], orig)
        h.remove()
        print(f"[selftest] live bf16 model, layer-4 MLP write along r: before {before:.3g}  after {after:.3g}; "
              f"re-applied on reload: {same}; revert restores the original bits: {restored}; cost {rec['cost']}")
        assert after < 0.05 * before and same and restored
        try:
            S.rank_one("model.embed_tokens.weight", torch.zeros(128000), torch.zeros(2048), rationale="must be rejected by the guard")
            raise AssertionError("embedding edit accepted")
        except GuardError:
            pass
        print("[selftest] all checks passed")
    finally:
        del S
        shutil.rmtree(ROOT / "results/surgery" / run, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new")
    p.add_argument("run")
    p.add_argument("--target", choices=("book", "calculus"), required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--note", default="")
    p.add_argument("--model", default=None)
    p = sub.add_parser("exec")
    p.add_argument("run")
    p.add_argument("step")
    p = sub.add_parser("status")
    p.add_argument("run")
    p = sub.add_parser("evalckpt")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="full")
    p.add_argument("--cfg", default=None)
    p = sub.add_parser("probe")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--target", choices=("book", "calculus"), default="calculus")
    p.add_argument("--steps", default="10,25,50")
    p.add_argument("--kind", default="mixed")
    p.add_argument("--n-items", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", default=None)
    p = sub.add_parser("cert")
    p.add_argument("run")
    p.add_argument("--uri", default=None)
    p = sub.add_parser("selftest")
    p.add_argument("--model", action="store_true")
    a = ap.parse_args()

    if a.cmd == "new":
        from lfm3u.surgery.tools import Session

        if (ROOT / "results/surgery" / a.run / "state.json").is_file():
            sys.exit(f"run {a.run} exists")
        S = Session(a.run, target=a.target, budget=a.budget, model_dir=a.model, note=a.note)
        show(S.guard.summary())
    elif a.cmd == "exec":
        cmd_exec(a)
    elif a.cmd == "status":
        from lfm3u.surgery.guard import Guard

        show(Guard(ROOT / "results/surgery" / a.run).summary(), indent=1)
    elif a.cmd == "evalckpt":
        cmd_evalckpt(a)
    elif a.cmd == "probe":
        cmd_probe(a)
    elif a.cmd == "cert":
        from lfm3u.surgery import cert

        print(cert.write(a.run, a.uri))
    elif a.cmd == "selftest":
        selftest(a.model)


if __name__ == "__main__":
    main()
