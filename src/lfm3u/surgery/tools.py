"""The surgery tools (PLAN 7.1): a Session wraps the target model, the guard and the dev scorer.

Read-only (not budgeted; they return weights, attribution scores or internal activations, never model outputs)
  list_params, tensor_stats, get_slice, attribution, activations, direction, jlens_readout
Budgeted (one forward call each)
  run_forward(prompts)   generations and/or teacher-forced log-probs for up to 256 prompts
  dev_score(tasks)       the dev scorer (score.py); stored as the before/after score of the edits around it
Edits (through the guard; each is a delta file, re-applied when the run is loaded in another process)
  ablate_direction, scale_units, rank_one, revert
save_patch writes the full bf16 checkpoint from the ORIGINAL weights plus the logged deltas and closes the run.

Weights are edited in fp32 (original + sum of active deltas) and cast to bf16, so edits commute and revert is exact.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from lfm3u.surgery import edits as E
from lfm3u.surgery import score
from lfm3u.surgery.guard import Guard, GuardError, edit_cost

ROOT = Path(__file__).resolve().parents[3]
TARGET_MODEL = ROOT / "models/target"
LENS = ROOT / "work/surgery/target_v3_lens.pt"
SCORES = {"book": ROOT / "results/localize/book/scores.pt", "calculus": ROOT / "results/localize/calculus/scores.pt"}


def log(msg: str) -> None:
    print(f"[surgery {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path: str | Path, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # On Windows the driver lets a process overflow into shared system memory instead of raising OOM, which
        # stalls the run and the desktop. Cap this process below the free dedicated memory so it fails loudly instead.
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.3, min(free - (1 << 30), int(0.8 * total)) / total))
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Session:
    def __init__(self, run: str, *, target: str | None = None, budget: int | None = None, model_dir: str | Path | None = None,
                 device: str | None = None, note: str = ""):
        import torch

        self.torch = torch
        self.run = run
        self.dir = ROOT / "results/surgery" / run
        state = self.dir / "state.json"
        if state.is_file():
            model_dir = json.loads(state.read_text(encoding="utf-8"))["meta"]["model_dir"]
        self.model_dir = Path(model_dir or TARGET_MODEL)
        self.model, self.tok = load_model(self.model_dir, device)
        self.device = next(self.model.parameters()).device
        self.params = dict(self.model.named_parameters())
        n_params = sum(p.numel() for p in self.model.parameters())
        self.guard = Guard(self.dir, n_params=n_params, call_budget=budget, target=target,
                           meta={"model_dir": str(self.model_dir), "note": note})
        self.target = self.guard.state["target"]
        (self.dir / "vecs").mkdir(exist_ok=True)
        self.orig: dict = {}       # fp32 CPU copies of the original values of edited tensors
        self._scores: dict = {}
        self._lens = None
        for name in sorted({e["param"] for e in self.guard.active()}):
            self._refresh(name)

    # ------------------------------------------------------------ weights = original + active deltas
    def _spec(self, rec: dict) -> dict:
        return self.torch.load(self.dir / rec["file"], weights_only=True)

    def effective(self, name: str):
        """fp32 value of a tensor on the device: original + all active deltas (the model holds its bf16 rounding)."""
        p = self.params[name]
        if name not in self.orig:
            return p.detach().float()
        w = self.orig[name].to(self.device)
        for e in self.guard.active():
            if e["param"] == name:
                w = w + E.dense(self._spec(e), p.shape, self.device)
        return w

    def _refresh(self, name: str) -> None:
        p = self.params[name]
        if name not in self.orig:
            self.orig[name] = p.detach().float().cpu()
        with self.torch.no_grad():
            p.copy_(self.effective(name).to(p.dtype))

    # ------------------------------------------------------------ read-only inspection
    def list_params(self, pattern: str = "") -> list[dict]:
        out = []
        for n, p in self.params.items():
            if pattern in n:
                out.append({"name": n, "shape": list(p.shape), "numel": p.numel(),
                            "editable": not any(k in n for k in ("embed_tokens", "lm_head", "embedding_norm"))})
        return out

    def tensor_stats(self, name: str, svd: int = 0) -> dict:
        w = self.effective(name)
        s = {"name": name, "shape": list(w.shape), "mean": float(w.mean()), "std": float(w.std()), "min": float(w.min()),
             "max": float(w.max()), "fro": float(w.norm()), "abs_mean": float(w.abs().mean())}
        if w.ndim == 2:
            s["row_norm_mean"], s["col_norm_mean"] = float(w.norm(dim=1).mean()), float(w.norm(dim=0).mean())
            if svd:
                s["singular_values"] = [round(float(x), 4) for x in self.torch.linalg.svdvals(w)[:svd]]
        return s

    def get_slice(self, name: str, rows=None, cols=None):
        w = self.effective(name)
        if w.ndim == 1:
            return w[rows].cpu() if rows is not None else w.cpu()
        w = w[rows] if rows is not None else w
        return (w[:, cols] if cols is not None else w).cpu()

    def attribution(self, target: str | None = None, top_k: int = 50, kinds=("mlp", "head", "conv"), layers=None,
                    lam: float = 1.0) -> dict:
        """Phase-5 attribution scores (results/localize/<target>/scores.pt). Returns the top units by forget-specific
        score s_forget - lam * max(s_retain, 0) and the per-layer forget-specific mass."""
        from lfm3u.unlearn import localize

        target = target or self.target
        if target not in self._scores:
            self._scores[target] = self.torch.load(SCORES[target], weights_only=False)
        sc = self._scores[target]
        hd = sc["meta"]["head_dim"]
        units, by_layer = [], {}
        for n, e in sc["modules"].items():
            if e["kind"] not in kinds or (layers is not None and e["layer"] not in layers):
                continue
            f, r = localize.unit_scores(e, hd, "forget"), localize.unit_scores(e, hd, "retain")
            s = f - lam * r.clamp(min=0)
            by_layer[e["layer"]] = by_layer.get(e["layer"], 0.0) + float(s.clamp(min=0).sum())
            k = min(top_k, len(s))
            top = s.topk(k)
            units += [{"module": n, "layer": e["layer"], "kind": e["kind"], "unit": int(i), "score": float(v),
                       "forget": float(f[i]), "retain": float(r[i])} for v, i in zip(top.values, top.indices)]
        units.sort(key=lambda u: -u["score"])
        return {"target": target, "top": units[:top_k], "specific_mass_by_layer": {k: round(v, 4) for k, v in sorted(by_layer.items())}}

    def raw_scores(self, target: str | None = None) -> dict:
        target = target or self.target
        if target not in self._scores:
            self._scores[target] = self.torch.load(SCORES[target], weights_only=False)
        return self._scores[target]

    # ------------------------------------------------------------ activations (read-only)
    def activations(self, prompts: list[list[int]], sites: list[str], *, reduce: str = "mean", skip_bos: bool = True,
                    cov: bool = False, bs: int = 16) -> dict:
        """Internal activations at `sites`, reduced over the real tokens of all prompts.
          "resid"            residual stream after the embedding and after every layer -> [n_layers + 1, d]
          "in:<module>"      input of a module (e.g. in:model.layers.5.feed_forward.w2 = MLP hidden units)
          "out:<module>"     output of a module
        reduce: "mean" over tokens (all prompts pooled), or "last" = mean over prompts of the last token.
        cov=True also returns the uncentred second moment E[a a^T] per in:/out: site ("<site>#m2", fp32 on CPU)."""
        torch = self.torch
        pad = self.tok.pad_token_id
        mods = dict(self.model.named_modules())
        store, handles = {}, []
        for s in sites:
            if s == "resid":
                continue
            kind, name = s.split(":", 1)
            if kind == "in":
                handles.append(mods[name].register_forward_pre_hook(lambda _m, args, s=s: store.__setitem__(s, args[0])))
            else:
                handles.append(mods[name].register_forward_hook(
                    lambda _m, _a, out, s=s: store.__setitem__(s, out[0] if isinstance(out, tuple) else out)))
        sums, m2, count = {}, {}, 0
        try:
            for b in range(0, len(prompts), bs):
                chunk = prompts[b:b + bs]
                L = max(len(p) for p in chunk)
                ids = torch.full((len(chunk), L), pad, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, p in enumerate(chunk):
                    ids[j, :len(p)] = torch.tensor(p)
                    att[j, :len(p)] = 1
                ids, att = ids.to(self.device), att.to(self.device)
                with torch.no_grad():
                    out = self.model.model(input_ids=ids, attention_mask=att, output_hidden_states="resid" in sites, use_cache=False)
                sel = att.bool()
                if reduce == "last":
                    sel = torch.zeros_like(sel)
                    sel[torch.arange(len(chunk)), att.sum(1) - 1] = True
                elif skip_bos:
                    sel[:, 0] = False
                count += int(sel.sum())
                if "resid" in sites:
                    v = torch.stack([h[sel].float().sum(0) for h in out.hidden_states]).double()
                    sums["resid"] = sums.get("resid", 0) + v
                for s in store:
                    a = store[s][sel].float()
                    sums[s] = sums.get(s, 0) + a.sum(0).double()
                    if cov:
                        m2[s] = m2.get(s, 0) + (a.T @ a).double()
                store.clear()
        finally:
            for h in handles:
                h.remove()
        res = {s: (v / max(count, 1)).float().cpu() for s, v in sums.items()}
        res |= {f"{s}#m2": (v / max(count, 1)).float().cpu() for s, v in m2.items()}
        res["n_tokens"] = count
        return res

    def resid_tokens(self, prompts: list[list[int]], layer: int, bs: int = 16):
        """Read-only. Per-token residual after `layer` for all real tokens except BOS: [N, d] fp32 on CPU. Used to
        measure how much an edit perturbs internal processing on forget vs retain text (`perturbation`)."""
        torch = self.torch
        pad, outs = self.tok.pad_token_id, []
        for b in range(0, len(prompts), bs):
            chunk = prompts[b:b + bs]
            L = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, p in enumerate(chunk):
                ids[j, :len(p)] = torch.tensor(p)
                att[j, :len(p)] = 1
            with torch.no_grad():
                hs = self.model.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device), output_hidden_states=True,
                                      use_cache=False).hidden_states
            sel = att.bool()
            sel[:, 0] = False
            outs.append(hs[layer + 1][sel.to(self.device)].float().cpu())
        return torch.cat(outs)

    def perturbation(self, base, prompts: list[list[int]], layer: int) -> float:
        """Read-only. Mean over tokens of |h' - h| / |h| at `layer`, h = `base` (resid_tokens of the unedited model on
        the same prompts), h' = the current weights. An internal measure of how much the active edits change the
        processing of this text; it is not a model output."""
        cur = self.resid_tokens(prompts, layer)
        return float(((cur - base).norm(dim=1) / base.norm(dim=1)).mean())

    def direction(self, forget: list[list[int]], contrast: list[list[int]], layer: int, *, name: str | None = None,
                  reduce: str = "mean") -> dict:
        """Unit mean-difference direction of the layer-`layer` output residual, forget minus contrast (as in
        scripts/11_abliterate.py). Saved as vecs/<name>.pt when a name is given."""
        mf = self.activations(forget, ["resid"], reduce=reduce)["resid"][layer + 1]
        mc = self.activations(contrast, ["resid"], reduce=reduce)["resid"][layer + 1]
        d = mf - mc
        r = d / d.norm()
        if name:
            self.save_vec(name, r)
        return {"r": r, "diff_norm": float(d.norm()), "forget_norm": float(mf.norm()), "contrast_norm": float(mc.norm())}

    def contrast_subspace(self, modules: list[str], forget: list[list[int]], retain: list[list[int]], *, k: int = 32,
                          shrink: float = 0.05, name: str | None = None, forget_val=None, retain_val=None) -> dict:
        """Read-only. For each writer module (<= 6 per call), the directions of its INPUT space (MLP hidden units, head
        outputs, conv channels) that carry much more energy on forget text than on retain text: generalized
        eigenvectors of (M_f, C), M = uncentred second moment of the input activations, C = M_r + shrink * mean
        diag(M_r) * I. Returns the top-k eigenvalues (forget / retain energy ratio) per module; with a name, saves
        vecs/<name>.<layer>.pt = {"V": [d, k] (C-orthonormal), "CV": C @ V, "evals"}. Removing the top-k components
        is the rank-k edit W += (-W V)(C V)^T (`ablate_subspace`): a = sum_i v_i (C v_i)^T a in this basis."""
        torch = self.torch
        sites = [f"in:{m}" for m in modules]
        af = self.activations(forget, sites, cov=True)
        ar = self.activations(retain, sites, cov=True)
        out = {"n_forget_tokens": af["n_tokens"], "n_retain_tokens": ar["n_tokens"], "modules": {}}
        # held-out energy ratios: the top eigenvalues of a 7,680-d problem fitted on ~1e5 tokens are inflated
        vf = self.activations(forget_val, sites, cov=True) if forget_val else None
        vr = self.activations(retain_val, sites, cov=True) if retain_val else None
        for m, s in zip(modules, sites):
            Mf = af[f"{s}#m2"].to(self.device, torch.float64)
            Mr = ar[f"{s}#m2"].to(self.device, torch.float64)
            C = Mr + shrink * Mr.diagonal().mean() * torch.eye(Mr.shape[0], device=self.device, dtype=torch.float64)
            Lc = torch.linalg.cholesky(C)
            A = torch.linalg.solve_triangular(Lc, Mf, upper=False)
            A = torch.linalg.solve_triangular(Lc, A.T, upper=False).T
            ev, w = torch.linalg.eigh((A + A.T) / 2)
            ev, w = ev.flip(0)[:k], w.flip(1)[:, :k]
            V = torch.linalg.solve_triangular(Lc.T, w, upper=True)
            CV = C @ V
            out["modules"][m] = {"evals": [round(float(x), 2) for x in ev], "trace_ratio": round(float(Mf.diagonal().sum() / Mr.diagonal().sum()), 3),
                                 "forget_energy_in_topk": round(float((V.T @ Mf @ CV).diagonal().sum() / Mf.diagonal().sum()), 4),
                                 "retain_energy_in_topk": round(float((V.T @ Mr @ CV).diagonal().sum() / Mr.diagonal().sum()), 4)}
            if vf is not None and vr is not None:
                Mfv, Mrv = vf[f"{s}#m2"].to(self.device, torch.float64), vr[f"{s}#m2"].to(self.device, torch.float64)
                num = ((V.T @ Mfv) * V.T).sum(1)
                den = ((V.T @ Mrv) * V.T).sum(1) + shrink * Mrv.diagonal().mean() * (V * V).sum(0)
                out["modules"][m]["evals_heldout"] = [round(float(x), 2) for x in num / den]
                del Mfv, Mrv
            if name:
                layer = m.split(".")[2]
                torch.save({"V": V.float().cpu(), "CV": CV.float().cpu(), "evals": ev.float().cpu(), "module": m, "shrink": shrink},
                           self.dir / "vecs" / f"{name}.{layer}.pt")
            del Mf, Mr, C, Lc, A, w, V, CV
        return out

    def save_vec(self, name: str, v) -> None:
        self.torch.save(v.detach().float().cpu(), self.dir / "vecs" / f"{name}.pt")

    def vec(self, name_or_tensor):
        if isinstance(name_or_tensor, str):
            return self.torch.load(self.dir / "vecs" / f"{name_or_tensor}.pt", weights_only=True)
        return name_or_tensor.detach().float().cpu()

    # ------------------------------------------------------------ J-lens (read-only: lens layers only, not the output)
    def jlens_readout(self, prompt: list[int], *, positions=(-1,), layers=None, k: int = 8, targets: dict | None = None) -> dict:
        """Jacobian-lens top-k tokens per lens layer at `positions` (target lens target_v3). `targets` = {label:
        [token ids]} adds the per-layer rank (min over ids). The model's real output distribution is NOT returned:
        that is what run_forward is budgeted for."""
        torch = self.torch
        from jlens import JacobianLens

        from lfm3u.jspace import readout
        from lfm3u.jspace.fit import wrap

        lens, jm = self._load_lens()
        final = jm.n_layers - 1
        lg = readout.layer_logits(jm, lens, torch.tensor([prompt]), positions=positions, layers=layers)
        out = {}
        for layer in sorted(lg):
            if layer == final:
                continue
            row = {"top": [[self.tok.decode([int(t)]) for t in lg[layer][p].topk(k).indices] for p in range(len(positions))]}
            if targets:
                row["rank"] = {lab: readout.ranks_from_logits(lg[layer], ids).tolist() for lab, ids in targets.items()}
            out[layer] = row
        return out

    def _load_lens(self):
        from jlens import JacobianLens

        from lfm3u.jspace import readout
        from lfm3u.jspace.fit import wrap

        if self._lens is None:
            lens = JacobianLens.load(str(LENS))
            readout.lens_to(lens, self.device)
            self._lens = (lens, wrap(self.model, self.tok))
        return self._lens

    def jlens_next_rank(self, seq: list[int], positions: list[int]) -> dict:
        """Read-only. For a given text (token ids), the lens rank of the TRUE next token at each position and lens
        layer: {layer: [rank per position]}. Shows at which depth a memorised continuation is already decided.
        Lens layers only (the model's own output rank is not returned)."""
        torch = self.torch
        from lfm3u.jspace import readout

        lens, jm = self._load_lens()
        final = jm.n_layers - 1
        lg = readout.layer_logits(jm, lens, torch.tensor([seq]), positions=positions)
        nxt = torch.tensor([seq[p + 1] for p in positions], device=self.device)
        out = {}
        for layer in sorted(lg):
            if layer != final:
                out[layer] = ((lg[layer] > lg[layer].gather(1, nxt[:, None])).sum(1) + 1).tolist()
        return out

    def jlens_direction(self, words: list[str], layer: int, name: str | None = None):
        """Read-only. J-space concept direction at a layer: v = J_l^T u_t, mean over the first-token variants of
        `words` (unit norm), as in lfm3u.jspace.inject.concept_direction. This is the residual direction at layer l
        that the lens reads as those tokens. Note: the lens was fitted on the UNEDITED target."""
        from lfm3u.jspace import inject, readout

        lens, jm = self._load_lens()
        ids = sorted({t for w in words for t in readout.first_token_variants(self.tok, w)})
        v = inject.concept_direction(lens, jm, ids, layer).float().cpu()
        if name:
            self.save_vec(name, v)
        return {"v": v, "token_ids": ids, "tokens": [self.tok.decode([t]) for t in ids]}

    # ------------------------------------------------------------ activation ablation (the hook baseline; not an edit)
    def hook_ablation(self, direction, *, embed: bool = True, layers=None):
        """Context manager: project r out of the residual stream after the embedding and after every layer
        (scripts/11_abliterate.py). Used only to compare the weight-space version with the hook."""
        import contextlib

        r = self.vec(direction).to(self.device)
        r = r / r.norm()

        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            h32 = h.float()
            h2 = (h32 - (h32 @ r).unsqueeze(-1) * r).to(h.dtype)
            return (h2, *output[1:]) if isinstance(output, tuple) else h2

        @contextlib.contextmanager
        def ctx():
            mods = [self.model.model.embed_tokens] if embed else []
            mods += [m for i, m in enumerate(self.model.model.layers) if layers is None or i in layers]
            hs = [m.register_forward_hook(hook) for m in mods]
            try:
                yield
            finally:
                for h in hs:
                    h.remove()

        return ctx()

    # ------------------------------------------------------------ budgeted forwards
    def run_forward(self, prompts: list[list[int]], *, max_new_tokens: int = 0, targets: list[list[int]] | None = None,
                    top_k: int = 5, note: str = "") -> list[dict]:
        """One budgeted call. Per prompt: greedy continuation (if max_new_tokens > 0), the top-k next tokens with
        log-probs, and the mean log-prob of `targets[i]` as a continuation (if given)."""
        torch = self.torch
        self.guard.charge("run_forward", len(prompts), note)
        res = [{} for _ in prompts]
        with torch.no_grad():
            for i, p in enumerate(prompts):
                h = score.hidden(self.model, torch.tensor([p], device=self.device), None)[0, -1]
                lp = torch.log_softmax(self.model.lm_head(h).float(), -1)
                top = lp.topk(top_k)
                res[i]["next"] = [[self.tok.decode([int(t)]), round(float(v), 3)] for v, t in zip(top.values, top.indices)]
        if targets is not None:
            lps = score.seq_logprobs(self.model, self.tok, [p + t for p, t in zip(prompts, targets)], [len(p) for p in prompts])
            for r, x in zip(res, lps):
                r["target_logprob"] = round(sum(x) / max(len(x), 1), 4)
        if max_new_tokens > 0:
            for r, o in zip(res, score.generate(self.model, self.tok, prompts, max_new_tokens)):
                r |= o
        return res

    def dev_score(self, tasks="fast", cfg: dict | None = None, note: str = "") -> dict:
        """One budgeted call: the dev scorer on the current weights. Returns the brief view; the full result is in
        scores/<call>.json and the generations in generations_<call>/."""
        n = self.guard.charge("dev_score", 0, note or str(tasks))
        res = score.evaluate(self.model, self.tok, tasks, cfg, self.dir, tag=f"_{n:02d}")
        (self.dir / "scores").mkdir(exist_ok=True)
        (self.dir / "scores" / f"{n:02d}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        self.guard.record_score({"tasks": tasks if isinstance(tasks, str) else list(tasks), "brief": res["brief"]})
        return res["brief"]

    # ------------------------------------------------------------ edits
    def _commit(self, param: str, spec: dict, *, op: str, args: dict, rationale: str, cost: int) -> dict:
        p = self.params.get(param)
        if p is None:
            raise GuardError(f"no parameter {param}")
        rec = self.guard.admit(param=param, shape=tuple(p.shape), kind=spec["kind"], cost=cost, op=op, args=args, rationale=rationale)
        try:
            if param not in self.orig:
                self.orig[param] = p.detach().float().cpu()
            self.torch.save({k: (v.detach().float().cpu() if k in ("U", "V", "delta") else v.cpu() if hasattr(v, "cpu") else v)
                             for k, v in spec.items()}, self.dir / rec["file"])
        except Exception:
            self.guard.drop(rec["id"])
            raise
        self._refresh(param)
        return {"id": rec["id"], "param": param, "op": op, "cost": cost, "edit_cost_used": self.guard.used(),
                "edit_budget": self.guard.state["edit_budget"]}

    def ablate_direction(self, param: str, direction, *, axis: str = "out", sandwich: bool = True, mode: str = "pre",
                         mass: float | None = None, kappa: float = 3.0, rationale: str = "") -> dict:
        """Remove a residual-stream direction r from a writer (axis="out": its post-norm write along r becomes zero)
        or blind a reader to it (axis="in"). Writer modes: "pre" = projection orthogonal in pre-norm space (PLAN
        correction 2), "stream" = projection orthogonal in stream space (the exact weight form of the activation
        hook; `kappa` or `mass` set which output dims are used). sandwich=False gives the naive projection (comparison only)."""
        self.guard.check_param(param)
        W = self.effective(param)
        r = self.vec(direction).to(self.device)
        if axis == "out":
            gname = E.writer_norm(param)
            if sandwich and gname is None:
                raise GuardError(f"{param} is not a residual-stream writer; pass sandwich=False for a plain projection")
            if sandwich and mode == "stream":
                U, V = E.ablate_out_stream(W, r, self.effective(gname), mass, kappa)
            else:
                U, V = E.ablate_out(W, r, self.effective(gname) if sandwich else None)
        elif axis == "in":
            gname = E.reader_norm(param)
            if sandwich and gname is None:
                raise GuardError(f"{param} is not a residual-stream reader; pass sandwich=False for a plain projection")
            g = self.effective(gname) if sandwich else None
            if g is not None and float(g.abs().min()) < 1e-4:
                raise GuardError(f"{gname} has near-zero gains; the reader-side edit would blow up")
            U, V = E.ablate_in(W, r, g)
        else:
            raise GuardError("axis must be 'out' or 'in'")
        args = {"axis": axis, "sandwich": sandwich, "mode": mode if axis == "out" else None, "kappa": kappa if mode == "stream" and mass is None else None, "mass": mass, "norm": gname if sandwich else None,
                "direction": direction if isinstance(direction, str) else "tensor", "delta_fro": float((U @ V.T).norm()),
                "rel_change": float((U @ V.T).norm() / W.norm())}
        return self._commit(param, {"kind": "lowrank", "U": U, "V": V}, op="ablate_direction", args=args, rationale=rationale,
                            cost=edit_cost("lowrank", tuple(W.shape), rank=1))

    def scale_units(self, module: str, idx: list[int], factor: float, *, rationale: str = "") -> dict:
        """Multiply the write columns of units by `factor`. `module` is a writer as in the localizer
        (model.layers.L.feed_forward.w2 | .self_attn.o_proj | .conv.out_proj); a head is 128 columns of o_proj."""
        param = module if module.endswith(".weight") else f"{module}.weight"
        self.guard.check_param(param)
        W = self.effective(param)
        width = self.model.config.head_dim if param.endswith("self_attn.o_proj.weight") else 1
        cols = self.torch.tensor(sorted({c for i in idx for c in range(i * width, (i + 1) * width)}), dtype=self.torch.long)
        delta = E.scale_cols(W, cols.to(self.device), factor)
        args = {"units": sorted(idx), "factor": factor, "n_cols": len(cols), "rel_change": float(delta.norm() / W.norm())}
        return self._commit(param, {"kind": "cols", "cols": cols, "delta": delta}, op="scale_units", args=args, rationale=rationale,
                            cost=edit_cost("cols", tuple(W.shape), n_cols=len(cols)))

    def rank_one(self, param: str, u, v, *, rationale: str = "", label: str = "") -> dict:
        """W += u v^T (u: [rows] or [rows, k]; v: [cols] or [cols, k]); cost k * (rows + cols)."""
        self.guard.check_param(param)
        W = self.effective(param)
        U, V = self.vec(u).to(self.device), self.vec(v).to(self.device)
        U, V = (U[:, None] if U.ndim == 1 else U), (V[:, None] if V.ndim == 1 else V)
        if U.shape[0] != W.shape[0] or V.shape[0] != W.shape[1] or U.shape[1] != V.shape[1]:
            raise GuardError(f"shape mismatch: W {tuple(W.shape)}, U {tuple(U.shape)}, V {tuple(V.shape)}")
        args = {"rank": U.shape[1], "label": label, "delta_fro": float((U @ V.T).norm()), "rel_change": float((U @ V.T).norm() / W.norm())}
        return self._commit(param, {"kind": "lowrank", "U": U, "V": V}, op="rank_one", args=args, rationale=rationale,
                            cost=edit_cost("lowrank", tuple(W.shape), rank=U.shape[1]))

    def ablate_subspace(self, module: str, sub: str, k: int, *, rationale: str = "", random_control: int | None = None) -> dict:
        """Rank-k edit that removes the top-k contrast_subspace components from a writer's input:
        W += (-W V_k)(C V_k)^T. random_control=seed removes k random orthonormal input directions instead
        (specificity control). Logged as a rank_one edit of rank k."""
        torch = self.torch
        param = module if module.endswith(".weight") else f"{module}.weight"
        d = torch.load(self.dir / "vecs" / f"{sub}.{param.split('.')[2]}.pt", weights_only=True)
        if d["module"] + ".weight" != param:
            raise GuardError(f"subspace {sub} was computed for {d['module']}")
        V, CV = d["V"][:, :k], d["CV"][:, :k]
        label = f"contrast subspace {sub}, top {k}"
        if random_control is not None:
            g = torch.Generator().manual_seed(random_control)
            Q, _ = torch.linalg.qr(torch.randn(V.shape[0], k, generator=g))
            V, CV = Q, Q
            label = f"random orthonormal subspace (seed {random_control}), rank {k}"
        W = self.effective(param)
        return self.rank_one(param, -(W @ V.to(self.device)), CV, rationale=rationale, label=label)

    def revert(self, edit_id: str, reason: str = "") -> dict:
        rec = self.guard.revert(edit_id, reason)
        self._refresh(rec["param"])
        return {"reverted": edit_id, "edit_cost_used": self.guard.used()}

    # ------------------------------------------------------------ patch
    def save_patch(self, name: str | None = None, note: str = "") -> dict:
        """Write models/surgery/<name>/ = original checkpoint + active deltas (fp32 sum, cast to bf16), verify it against
        the in-memory model, and close the run."""
        torch = self.torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        if self.guard.state["closed"]:
            raise GuardError("run already closed")
        name = name or self.run
        out = ROOT / "models/surgery" / name
        out.mkdir(parents=True, exist_ok=True)
        src = self.model_dir / "model.safetensors"
        tensors, changed = {}, {}
        active = self.guard.active()
        with safe_open(str(src), framework="pt") as f:
            meta = f.metadata() or {"format": "pt"}
            keys = list(f.keys())
            for k in keys:
                t = f.get_tensor(k)
                recs = [e for e in active if e["param"] == k]
                if recs:
                    w = t.float()
                    for e in recs:
                        w = w + E.dense(self._spec(e), t.shape)
                    new = w.to(t.dtype)
                    mem = self.params[k].detach().cpu()
                    # the in-memory copy was summed on the GPU: allow last-bit rounding differences, nothing more
                    gap = float((new.float() - mem.float()).abs().max())
                    if gap > 1e-3 * float(new.float().abs().max()):
                        raise GuardError(f"{k}: checkpoint value differs from the in-memory model (max gap {gap:.3g})")
                    changed[k] = {"elements": int((new != t).sum()), "numel": t.numel(),
                                  "max_abs_delta": float((new.float() - t.float()).abs().max()),
                                  "elements_differing_from_memory": int((new != mem).sum())}
                    t = new
                tensors[k] = t
        missing = sorted({e["param"] for e in active} - set(keys))
        if missing:
            raise GuardError(f"edited tensors not in the checkpoint: {missing}")
        save_file(tensors, str(out / "model.safetensors"), metadata=meta)
        del tensors
        for f in self.model_dir.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy2(f, out / f.name)
        g = self.guard
        patch = {"dir": str(out.relative_to(ROOT).as_posix()), "name": name, "note": note, "base_model": str(self.model_dir),
                 "sha256_before": sha256(src), "sha256_after": sha256(out / "model.safetensors"),
                 "n_params": g.state["n_params"], "edit_budget": g.state["edit_budget"], "lowrank_param_count": g.used(),
                 "lowrank_frac": g.used() / g.state["n_params"],
                 "changed_elements": sum(c["elements"] for c in changed.values()),
                 "changed_frac": sum(c["elements"] for c in changed.values()) / g.state["n_params"],
                 "changed_tensors": changed, "n_edits": len(active), "calls_used": g.state["calls_used"],
                 "call_budget": g.state["call_budget"]}
        (out / "lfm3u_meta.json").write_text(json.dumps({"name": name, "method": "surgery", "target": self.target, "surgery": patch,
                                                           "edits": active}, indent=1), encoding="utf-8")
        g.close(patch)
        return {k: v for k, v in patch.items() if k != "changed_tensors"}
