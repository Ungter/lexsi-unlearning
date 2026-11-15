"""Phase 7 driver: agentic weight surgery (src/lfm3u/surgery). State persists in results/surgery/<run>/.

  new RUN --target book|calculus --budget 10|25|50 [--note TEXT]   create a run (forward budget is fixed here)
  exec RUN STEP.py        run a step file with `S` (the Session) in scope; the file and its output are copied to
                          results/surgery/RUN/steps/NN_<name>.{py,out} as the agent's trace
  status RUN              budgets used and active edits
  evalckpt DIR --out OUT [--tasks full] [--cfg JSON]    dev eval of a saved checkpoint (not budgeted; the audit side)
  probe DIR --out OUT [--target calculus|book] [--steps 10,25,50] [--kind mixed|diff|integral|defint] [--cfg JSON]
                          relearning probe (lfm3u.surgery.relearn) on a saved checkpoint: not budgeted, the audit side;
                          writes OUT/probe_<kind>_s<seed>.json and the generations per step
  suite DIR --out OUT [--k 6] [--steps 10,25,50,100] [--draw-sha DIR2 | --draw LABEL | --configs FILE] [--tag NAME] [--dry]
                          P6 probe suite (K sampled attack configurations, lfm3u.surgery.relearn.suite) on a saved
                          checkpoint: not budgeted, the audit side. The draw seed comes from the SHA-256 of
                          DIR2/model.safetensors (default DIR itself) or from LABEL, or the configurations of a stored
                          suite result FILE are replayed. Writes OUT/<tag>.json and OUT/<tag>/cfg<i>/step_<k>.jsonl
  cert RUN [--uri s3://...]                             write results/surgery/RUN/certificate.md
  selftest [--model]      numeric checks of the edit math and the guard (--model adds the checks on the real model,
                          including that the relearning probe leaves every weight bit-identical)

Step files are plain Python. In scope: S, torch, score (lfm3u.surgery.score), show(obj) (prints JSON), ROOT.

Model facts for the agent (Tiny Aya Global, CohereLabs/tiny-aya-global, Cohere2ForCausalLM; see src/lfm3u/arch.py):
  36 decoder layers model.layers.0..35, every one attention (3 sliding-window 4096 : 1 full/NoPE), d_model 2048,
  16 query / 4 KV heads of 128, SwiGLU MLP 11008, vocab 262,144, 3.35B parameters, 8K context, no thinking mode.
  PARALLEL blocks: h = x + self_attn(ln(x)) + mlp(ln(x)) with ONE input_layernorm (LayerNorm, weight only, no bias);
  there is no mid-block residual, no post-norm and no ffn_norm. Residual writers: self_attn.o_proj [2048, 2048]
  (input = 16 heads x 128) and mlp.down_proj [2048, 11008] (input = MLP neurons); both write the same residual.
  Readers: self_attn.{q,k,v}_proj and mlp.{gate,up}_proj, all reading input_layernorm(x). The final norm is
  model.norm; embeddings are tied (lm_head = model.embed_tokens); both are protected. Logits = lm_head(h) *
  config.logit_scale. LFM3 names map as w1 -> mlp.gate_proj, w3 -> mlp.up_proj, w2 -> mlp.down_proj,
  self_attn.out_proj -> self_attn.o_proj, embedding_norm -> model.norm; conv.* has no counterpart. LFM3 layer indices
  (24 layers) scale by 1.5.
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


def cmd_suite(a) -> None:
    from lfm3u.surgery import relearn
    from lfm3u.surgery.tools import load_model, sha256_files, weight_files

    out = Path(a.out)
    steps = [int(s) for s in a.steps.split(",")]
    if a.configs:
        configs, draw = json.loads(Path(a.configs).read_text(encoding="utf-8"))["configs"], {"replay_of_file": str(a.configs)}
    else:
        # sha256 of model.safetensors, or of the shard list for a sharded checkpoint (tools.sha256_files)
        label = f"audit|{a.draw}" if a.draw else "audit|sha256|" + sha256_files(weight_files(Path(a.draw_sha or a.dir)))
        seed = relearn.seed_from(label, a.k)
        configs, draw = relearn.sample_configs(seed, a.k), {"label": label, "seed": seed}
    if a.dry:
        show({"draw": draw, "configs": [c["name"] for c in configs]}, indent=1)
        return
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(a.dir)
    res = relearn.suite(model, tok, configs, steps=steps, cfg=json.loads(a.cfg) if a.cfg else None, out_dir=out / a.tag)
    res |= {"model": str(a.dir), "charged": False, "draw": draw}
    (out / f"{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    show(relearn.suite_brief(res), indent=1)
    print("SUITE DONE", flush=True)


def selftest(with_model: bool) -> None:
    import shutil
    import tempfile

    import torch

    from lfm3u.surgery import edits as E
    from lfm3u.surgery.guard import BudgetExceeded, Guard, GuardError, edit_cost

    torch.manual_seed(0)
    dd = torch.float64
    # 1. writer side (Tiny Aya: no post-norm, the write W a goes straight into the stream): (I - r r^T) W zeroes the
    #    write along r; the LFM3 "stream" variant with g=None is the same edit; writer_norm is None for both writers
    rows, cols = 64, 200
    W = torch.randn(rows, cols, dtype=dd)
    r = E.unit(torch.randn(rows, dtype=dd))
    A = torch.randn(500, cols, dtype=dd)

    def write_along_r(Wx):
        z = A @ Wx.T
        return float(((z @ r).abs() / z.norm(dim=-1)).max())

    U, V = E.ablate_out(W, r, None)
    Us, Vs = E.ablate_out_stream(W, r, None)
    good, before = write_along_r(W + U @ V.T), write_along_r(W)
    assert good < 1e-12, good
    assert torch.allclose(U @ V.T, Us @ Vs.T)
    assert E.is_writer("model.layers.3.mlp.down_proj.weight") and E.is_writer("model.layers.3.self_attn.o_proj.weight")
    assert E.writer_norm("model.layers.3.mlp.down_proj.weight") is None
    assert E.reader_norm("model.layers.3.mlp.up_proj.weight") == "model.layers.3.input_layernorm.weight"
    print(f"[selftest] writer: |z.r|/|z| max  unedited {before:.3g}  (I - rr^T)W {good:.3g}")
    # 2. reader side through a LayerNorm ln(x) = g_in * (x - mean) / std: with r centred, W' ln(x) = W g_in * P(x - mean) / std,
    #    P = I - r r^T, so the reader is blind to adding any multiple of r to x (std aside, which we hold fixed here)
    Wr = torch.randn(300, rows, dtype=dd)
    gi = torch.rand(rows, dtype=dd) * 0.3 + 0.02
    X = torch.randn(500, rows, dtype=dd)
    Xc = X - X.mean(-1, keepdim=True)
    rc = E.unit(E.center(r))
    U, V = E.ablate_in(Wr, rc, gi)
    lhs = (gi * Xc) @ (Wr + U @ V.T).T
    rhs = (gi * (Xc - (Xc @ rc)[:, None] * rc)) @ Wr.T
    err = float((lhs - rhs).abs().max() / rhs.abs().max())
    X2 = X + 3.0 * r                                         # uncentred r: its mean part is removed by the LayerNorm anyway
    lhs2 = (gi * (X2 - X2.mean(-1, keepdim=True))) @ (Wr + U @ V.T).T
    blind = float((lhs2 - lhs).abs().max() / lhs.abs().max())
    assert err < 1e-10 and blind < 1e-10, (err, blind)
    print(f"[selftest] reader (LayerNorm): max relative error {err:.3g}; change when r is added to x {blind:.3g}")
    # 3. guard
    assert edit_cost("lowrank", (2048, 11008), rank=1) == 13056 and edit_cost("cols", (2048, 11008), n_cols=3) == 6144
    tmp = Path(tempfile.mkdtemp(prefix="guardtest_", dir=str(ROOT / "results")))
    try:
        try:
            Guard(tmp / "r", n_params=1_000_000, call_budget=7)
            raise AssertionError("budget 7 accepted")
        except GuardError:
            pass
        gd = Guard(tmp / "r", n_params=1_000_000, call_budget=10, target="book")
        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight", "model.embedding_norm.weight"):
            try:
                gd.admit(param=name, shape=(4, 4), kind="lowrank", cost=8, op="rank_one", args={}, rationale="should be rejected")
                raise AssertionError(name)
            except GuardError:
                pass
        gd.admit(param="model.layers.0.mlp.down_proj.weight", shape=(10, 50), kind="lowrank", cost=600, op="rank_one", args={}, rationale="within budget")
        try:
            gd.admit(param="model.layers.1.mlp.down_proj.weight", shape=(10, 50), kind="lowrank", cost=600, op="rank_one", args={}, rationale="over budget")
            raise AssertionError("edit budget not enforced")
        except BudgetExceeded:
            pass
        gd.revert("e001", "test")
        gd.admit(param="model.layers.1.mlp.down_proj.weight", shape=(10, 50), kind="lowrank", cost=600, op="rank_one", args={}, rationale="fits after revert")
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
            Guard(tmp / "r").admit(param="model.layers.2.mlp.down_proj.weight", shape=(10, 50), kind="lowrank", cost=1, op="rank_one", args={}, rationale="closed run")
            raise AssertionError("closed run accepted an edit")
        except GuardError:
            pass
        print("[selftest] guard: protected tensors, edit budget, revert, prompt cap, persisted forward counter, probe cost, fixed budget, close: ok")
        # P6 suite: budget 100 exists, a suite is charged K * 3 as one charge and refused as a whole; draws are deterministic
        # in the seed and stratified (one configuration per family and per data size in every block of three slots)
        from lfm3u.surgery import relearn

        g3 = Guard(tmp / "s", n_params=1_000_000, call_budget=100, target="calculus")
        assert relearn.suite_cost(3, (25, 100)) == 9 and relearn.suite_cost(6, (10, 25, 50, 100)) == 30
        assert g3.charge("relearn_suite", 3, cost=9) == 9
        for bad in (2, 92):
            try:
                g3.charge("relearn_suite", 3, cost=bad)
                raise AssertionError(f"suite charge {bad} accepted")
            except GuardError:
                pass
        assert g3.state["calls_used"] == 9
        for sd in range(200):
            cs = relearn.sample_configs(sd, 6)
            assert cs == relearn.sample_configs(sd, 6) and cs[:3] == relearn.sample_configs(sd, 3)
            for blk in (cs[:3], cs[3:]):
                assert sorted(c["family"] for c in blk) == ["full", "lora16", "lorax"] and sorted(c["n_items"] for c in blk) == [20, 60, 133]
            assert sorted(c["lr"] for c in cs if c["family"] == "full") == [1e-5, 3e-5]
            assert sorted(c["lr"] for c in cs if c["family"] == "lora16") == [2e-4, 1e-3]
            assert len({(c["rank"], c["targets"]) for c in cs if c["family"] == "lorax"}) == 2
        seen = {(c["family"], c["lr"], c["rank"], c["targets"], c["n_items"], c["schedule"]) for sd in range(1000) for c in relearn.sample_configs(sd, 3)}
        assert len(seen) == (2 + 2 + 8) * 3 * 2, len(seen)
        print(f"[selftest] suite: cost rule, whole-charge refusal, deterministic stratified draws, all {len(seen)} cells reachable: ok "
              f"(distribution hash {relearn.suite_hash()[:12]})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not with_model:
        return
    # 4. real model: real weights; bf16 rounding; persistence; revert. Layer 6 = LFM3's layer 4 scaled by 1.5.
    from lfm3u.surgery import score
    from lfm3u.surgery.tools import Session

    run = "_selftest"
    shutil.rmtree(ROOT / "results/surgery" / run, ignore_errors=True)
    S = Session(run, target="calculus", budget=25, note="selftest")
    try:
        tok = S.tok
        fp = [score.chat_ids(tok, f"Find the derivative of f(x) = {e}.") for e in ("x^3 + 2*x", "sin(3*x)*exp(x)", "log(x^2 + 1)", "x*cos(x)")]
        cp = [score.chat_ids(tok, q) for q in ("Compute 17 * 23.", "Solve 3*x + 5 = 20 for x.", "Simplify (x^2 - 1)/(x - 1).", "What is 144 / 12?")]
        L = 6
        r = S.direction(fp, cp, L, name="r4")["r"]
        layer = S.model.model.layers[L]
        name = f"model.layers.{L}.mlp.down_proj.weight"
        W32 = S.effective(name)
        hid = S.activations(fp + cp, [f"in:model.layers.{L}.mlp.down_proj"], cov=True)
        a = torch.randn(256, W32.shape[1], device=S.device) * hid[f"in:model.layers.{L}.mlp.down_proj#m2"].diagonal().sqrt().to(S.device)
        rr = r.to(S.device)

        def rel(Wx):
            z = a @ Wx.T                                        # no post-norm: this is the stream write
            return float(((z @ rr).abs() / z.norm(dim=-1)).mean())

        U, V = E.ablate_out(W32, rr, None)
        b0, b_good = rel(W32), rel(W32 + U @ V.T)
        b_bf16 = rel((W32 + U @ V.T).to(torch.bfloat16).float())
        print(f"[selftest] layer-{L} down_proj, mean |z.r|/|z|: unedited {b0:.3g}  (I - rr^T)W fp32 {b_good:.3g}  bf16 {b_bf16:.3g}")
        assert b_good < 1e-4 * b0 and b_bf16 < 0.02 * b0
        # in the running bf16 model: hook the down_proj output (= the MLP's stream write) on real prompts
        cap = {}
        h = layer.mlp.down_proj.register_forward_hook(lambda _m, _i, o: cap.__setitem__("z", o))

        def live():
            ids = torch.tensor([fp[1]], device=S.device)
            with torch.no_grad():
                S.model.model(input_ids=ids)
            z = cap["z"][0].float()
            return float(((z @ rr).abs() / z.norm(dim=-1)).mean())

        before = live()
        orig = S.params[name].detach().clone()
        rec = S.ablate_direction(name, "r4", rationale="selftest: remove r4 from the layer-6 MLP write")
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
        # max_len 1024: Tiny Aya prompts carry a ~350-token system preamble, so 384 cut every answer (no labels, loss 0)
        pr = S.relearn_probe(steps=(0, 2), n_items=4, max_len=1024, cfg={"calc_per_level": 1, "calc_max_new": 48}, note="selftest")
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
        gn = f"model.layers.{L}.input_layernorm.weight"
        g0 = S.params[gn].detach().clone()
        rec2 = S.set_elements(gn, [3, 17], 0.0, rationale="selftest: close two input coordinates of the layer-6 attention and MLP")
        zeroed = bool((S.params[gn][[3, 17]] == 0).all()) and int((S.params[gn] != g0).sum()) <= 2 and rec2["cost"] == 2
        snap = {n: p.detach().cpu().clone() for n, p in S.model.named_parameters()}
        pr2 = S.relearn_probe(steps=(1,), n_items=4, max_len=1024, gains=True, cfg={"calc_per_level": 1, "calc_max_new": 48}, note="selftest")
        now = dict(S.model.named_parameters())
        identical2 = list(now) == list(snap) and all(torch.equal(p.detach().cpu(), snap[n]) for n, p in now.items())
        del snap
        moved = S.guard.state["probes"][-1]["gains"]
        S.revert(rec2["id"], "selftest")
        back = torch.equal(S.params[gn], g0)
        try:
            S.set_elements("model.norm.weight", [0], 0.0, rationale="must be rejected by the guard")
            raise AssertionError("final-norm edit accepted")
        except GuardError:
            pass
        print(f"[selftest] norm-gain edit: entries zeroed, cost 2: {zeroed}; probe with gain adapters ({moved['n_vectors']} vectors, "
              f"max |delta| {moved['max_abs_delta']}) leaves weights bit-identical: {identical2}; revert restores the gain: {back}; "
              f"final norm protected; scored steps {sorted(pr2)}")
        assert zeroed and identical2 and back and moved["max_abs_delta"] > 0
        # probe suite (P6), run while the edit is active: three sampled configurations (one all-weight fine-tune on an
        # fp32 copy, two LoRA on bf16 copies), charged 3 * 3 calls; the session model keeps every bit (full element-wise
        # comparison against a CPU copy), its dtype, its frozen state and its parameter objects; a replay reuses the draw
        snap = {n: p.detach().cpu().clone() for n, p in S.model.named_parameters()}
        used = S.guard.state["calls_used"]
        tiny = {"cfg": {"calc_per_level": 1, "calc_max_new": 48}, "_selftest": {"n_items": 4, "max_len": 1024}}
        sb = S.relearn_probe_suite(k=3, steps=(1,), note="selftest", **tiny)
        call = S.guard.state["calls_used"]
        full_res = json.loads((S.dir / "scores" / f"{call:02d}_suite.json").read_text(encoding="utf-8"))
        fams = sorted(p["config"]["family"] for p in full_res["per_config"])
        n_full = [p["n_trainable"] for p in full_res["per_config"] if p["config"]["full"]]
        sb2 = S.relearn_probe_suite(steps=(1,), note="selftest replay", replay_of=call, **tiny)
        now = dict(S.model.named_parameters())
        identical3 = list(now) == list(snap) and all(torch.equal(p.detach().cpu(), snap[n]) for n, p in now.items())
        clean3 = (all(now[n] is p for n, p in S.params.items()) and not any("lora" in n.lower() for n, _ in S.model.named_modules())
                  and not any(p.requires_grad for p in S.model.parameters()) and all(p.dtype == torch.bfloat16 for p in S.model.parameters())
                  and not S.model.training)
        charged3 = S.guard.state["calls_used"] - used
        del snap
        print(f"[selftest] probe suite (edit active): configurations {list(sb['configs'])}; charged {charged3} calls for two suites; "
              f"all-weight copy trained {n_full[0]:,} of {S.guard.state['n_params']:,} parameters; session weights bit-identical: "
              f"{identical3}; model clean (objects, dtype, frozen, no adapter): {clean3}; replay used the same draw: "
              f"{list(sb2['configs']) == list(sb['configs'])}; max over configurations {sb['max']}")
        assert identical3 and clean3 and charged3 == 18 and fams == ["full", "lora16", "lorax"] and n_full == [S.guard.state["n_params"]]
        assert list(sb2["configs"]) == list(sb["configs"]) and S.guard.state["meta"]["suite_hash"] == full_res["suite_hash"]
        S.revert(rec["id"], "selftest")
        restored = torch.equal(S.params[name], orig)
        h.remove()
        print(f"[selftest] live bf16 model, layer-{L} MLP write along r: before {before:.3g}  after {after:.3g}; "
              f"re-applied on reload: {same}; revert restores the original bits: {restored}; cost {rec['cost']}")
        assert after < 0.05 * before and same and restored
        try:
            emb = S.params["model.embed_tokens.weight"]
            S.rank_one("model.embed_tokens.weight", torch.zeros(emb.shape[0]), torch.zeros(emb.shape[1]), rationale="must be rejected by the guard")
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
    p = sub.add_parser("suite")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--steps", default="10,25,50,100")
    p.add_argument("--draw-sha", default=None, help="checkpoint dir whose model.safetensors SHA-256 seeds the draw (default: DIR)")
    p.add_argument("--draw", default=None, help="label that seeds the draw instead of a checkpoint hash")
    p.add_argument("--configs", default=None, help="stored suite result (json) whose configurations are replayed")
    p.add_argument("--tag", default="suite")
    p.add_argument("--cfg", default=None)
    p.add_argument("--dry", action="store_true", help="print the drawn configurations and exit")
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
    elif a.cmd == "suite":
        cmd_suite(a)
    elif a.cmd == "cert":
        from lfm3u.surgery import cert

        print(cert.write(a.run, a.uri))
    elif a.cmd == "selftest":
        selftest(a.model)


if __name__ == "__main__":
    main()
