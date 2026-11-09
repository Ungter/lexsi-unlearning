"""SafeTune smoke test on the MUSE target model (PLAN 4.1-4.4 "(verify)" items).

For each method x target it loads fresh weights, runs a few unlearning steps through the same SafeTune entry
points Phase 4 will use, and records what actually happened:

  ga        unlearn.GradientAscentTrainer (full weights, fp32 upcast)
  graddiff  unlearn.GradDiffTrainer       (full weights)
  npo       unlearn.NPOTrainer, variant npo_KL, LoRA on LFM3's modules via safetune.configure(lora_target_modules)
  rmu       low-level safetune.unlearn.rmu_unlearn with param_substring="feed_forward.w2" (RMUTrainer has no such
            argument); batches moved to the GPU by hand

Targets (frozen Phase-2 data, train/dev only):
  book      forget = book/raw_forget_train (raw text), retain = book/raw_retain1_train; probes on verbmem_forget_dev
            (prompt + continuation) and raw_retain1_dev
  calculus  forget = calculus/level1-3/train (QA, answer-only labels), retain = algebra/level1-3/train; probes on
            calculus/level1/dev and algebra/level1/dev

Per run: steps, seconds, peak GPU memory, finite weights, which parameters changed (by module kind), forget/retain
NLL before and after, and a greedy completion before/after. Writes <out>/smoke.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u.data.batches import qa_batches, raw_batches  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]
LAYER_TYPES_KEY = "layer_types"


def log(msg: str) -> None:
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data

def load_split(data: Path, target: str, tok, n: int, max_len: int, seed: int = 0):
    rng = random.Random(seed)

    def sample(rows):
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:n]

    if target == "book":
        forget = raw_batches(sample(read_jsonl(data / "book/raw_forget_train.jsonl")), tok, max_len)
        retain = raw_batches(sample(read_jsonl(data / "book/raw_retain1_train.jsonl")), tok, max_len)
        verb = sample(read_jsonl(data / "book/verbmem_forget_dev.jsonl"))[:8]
        probe_f = [_continuation(tok, r["prompt"], r["target"], max_len) for r in verb]
        probe_r = raw_batches(sample(read_jsonl(data / "book/raw_retain1_dev.jsonl"))[:8], tok, max_len)[:8]
        prompt_f = [tok.bos_token + verb[0]["prompt"][-400:]]
        prompt_r = [tok.bos_token + read_jsonl(data / "book/raw_retain1_dev.jsonl")[0]["text"][:400]]
        return forget, retain, probe_f, probe_r, ("raw", prompt_f, prompt_r)
    rows = lambda topic, split: [r for lvl in (1, 2, 3) for r in read_jsonl(data / f"{topic}/level{lvl}/{split}.jsonl")]  # noqa: E731
    forget = qa_batches(sample(rows("calculus", "train")), tok, max_len)
    retain = qa_batches(sample(rows("algebra", "train")), tok, max_len)
    dev_f = sample(read_jsonl(data / "calculus/level1/dev.jsonl"))[:8]
    dev_r = sample(read_jsonl(data / "algebra/level1/dev.jsonl"))[:8]
    probe_f, probe_r = qa_batches(dev_f, tok, max_len), qa_batches(dev_r, tok, max_len)
    return forget, retain, probe_f, probe_r, ("chat", [dev_f[0]["instruction"]], [dev_r[0]["instruction"]])


def _continuation(tok, prompt: str, target: str, max_len: int) -> dict:
    import torch

    t = tok(target, add_special_tokens=False)["input_ids"][:max_len // 2]
    p = [tok.bos_token_id] + tok(prompt, add_special_tokens=False)["input_ids"][-(max_len - len(t) - 1):]
    ids, lab = p + t, [-100] * len(p) + t
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), dtype=torch.long), "labels": torch.tensor(lab)}


# ------------------------------------------------------------------ measurement

def mean_nll(model, batches) -> float:
    import torch
    import torch.nn.functional as F

    dev = next(model.parameters()).device
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for b in batches:
            ids, lab = b["input_ids"][None].to(dev), b["labels"][None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(input_ids=ids).logits
            lp = F.cross_entropy(logits[0, :-1].float(), lab[0, 1:], ignore_index=-100, reduction="sum")
            total, count = total + float(lp), count + int((lab[0, 1:] != -100).sum())
    return round(total / max(count, 1), 4)


def complete(model, tok, prompts: list[str], mode: str, n: int = 40) -> list[str]:
    import torch

    dev = next(model.parameters()).device
    out = []
    model.eval()
    for p in prompts:
        if mode == "chat":
            text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        else:
            text = p
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            g = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                               top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        out.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    return out


def module_kind(name: str, layer_types: list[str]) -> str:
    parts = name.split(".")
    if "layers" in parts:
        i = int(parts[parts.index("layers") + 1])
        sub = next((s for s in ("feed_forward", "self_attn", "conv", "operator_norm", "ffn_norm") if s in parts), "other")
        return f"{layer_types[i] if i < len(layer_types) else '?'}:{sub}"
    return parts[-2] if len(parts) > 1 else name


def snapshot(model) -> dict:
    """Cheap per-tensor fingerprint (float64 sum of |w| and of w^2) to detect which tensors changed."""
    import torch

    with torch.no_grad():
        return {n: (float(p.detach().double().abs().sum()), float(p.detach().double().pow(2).sum()))
                for n, p in model.named_parameters()}


def changed(before: dict, after: dict, layer_types: list[str]) -> dict:
    kinds: dict[str, int] = {}
    names = []
    for n, v in after.items():
        key = n.replace("base_model.model.", "").replace(".base_layer", "")
        b = before.get(key) or before.get(n)
        if b is None or "lora_" in n:
            continue
        if any(abs(x - y) > 1e-9 * max(1.0, abs(y)) for x, y in zip(v, b)):
            names.append(key)
            k = module_kind(key, layer_types)
            kinds[k] = kinds.get(k, 0) + 1
    return {"n_tensors": len(names), "by_kind": dict(sorted(kinds.items())), "first": names[:6]}


# ------------------------------------------------------------------ methods

def run_method(method: str, model, tok, forget, retain, steps: int, model_id: str, rmu_layer: int = 7):
    import safetune
    from safetune.runner import unlearn

    if method == "ga":
        t = unlearn.GradientAscentTrainer(model, model_id=model_id, lr=1e-5, epochs=1, max_steps=steps, forget_clip=None)
        return t.unlearn(forget=forget[:steps], retain=retain[:steps])
    if method == "graddiff":
        t = unlearn.GradDiffTrainer(model, model_id=model_id, lr=1e-5, epochs=1, max_steps=steps)
        return t.unlearn(forget=forget[:steps], retain=retain[:steps])
    if method == "npo":
        safetune.configure(lora_target_modules=LORA_TARGETS)
        t = unlearn.NPOTrainer(model, model_id=model_id, variant="npo_KL", beta=0.1, num_epochs=1, lr=1e-5)
        return t.unlearn(forget=forget[:steps], retain=retain[:steps])
    if method == "rmu":
        import torch
        import safetune.unlearn as U

        model = model.float()
        dev = next(model.parameters()).device
        move = lambda bs: [{k: v[None].to(dev) for k, v in b.items()} for b in bs]  # noqa: E731
        cfg = U.RMUConfig(layer_id=rmu_layer, update_layer_ids=[rmu_layer - 2, rmu_layer - 1, rmu_layer],
                          param_substring="feed_forward.w2",
                          steering_coeff=20.0, alpha=100.0, lr=5e-5, max_num_batches=steps)
        U.rmu_unlearn(model, retain_batches=move(retain[:steps]), forget_batches=move(forget[:steps]), config=cfg)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("rmu: non-finite weights")
        return model
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/safetune_smoke")
    ap.add_argument("--methods", default="ga,graddiff,npo,rmu")
    ap.add_argument("--targets", default="book,calculus")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rmu-layer", type=int, default=7, help="RMU target layer; updates w2 of this layer and the two below")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import safetune

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model, "steps": a.steps, "max_len": a.max_len, "safetune": safetune.__version__,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "runs": []}
    data_cache = {}
    for target in a.targets.split(","):
        data_cache[target] = load_split(Path(a.data), target, tok, max(a.steps, 16), a.max_len)
        f, r, pf, pr, _ = data_cache[target]
        log(f"{target}: {len(f)} forget / {len(r)} retain batches, probes {len(pf)}/{len(pr)}")
    for target in a.targets.split(","):
        forget, retain, probe_f, probe_r, (mode, prompt_f, prompt_r) = data_cache[target]
        for method in a.methods.split(","):
            rec = {"method": method, "target": target}
            t0 = time.time()
            try:
                model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to(device)
                layer_types = list(getattr(model.config, LAYER_TYPES_KEY, []))
                rec["before"] = {"forget_nll": mean_nll(model, probe_f), "retain_nll": mean_nll(model, probe_r),
                                 "forget_sample": complete(model, tok, prompt_f, mode)[0],
                                 "retain_sample": complete(model, tok, prompt_r, mode)[0]}
                snap = snapshot(model)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                model.train()
                model = run_method(method, model, tok, forget, retain, a.steps, a.model, a.rmu_layer)
                rec["train_seconds"] = round(time.time() - t1, 1)
                rec["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if device == "cuda" else None
                rec["dtype_after"] = str(next(model.parameters()).dtype)
                rec["changed"] = changed(snap, snapshot(model), layer_types)
                rec["finite"] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
                rec["after"] = {"forget_nll": mean_nll(model, probe_f), "retain_nll": mean_nll(model, probe_r),
                                "forget_sample": complete(model, tok, prompt_f, mode)[0],
                                "retain_sample": complete(model, tok, prompt_r, mode)[0]}
                rec["ok"] = rec["finite"] and rec["changed"]["n_tensors"] > 0 and all(
                    math.isfinite(v) for v in (rec["after"]["forget_nll"], rec["after"]["retain_nll"]))
            except Exception as e:  # noqa: BLE001
                rec |= {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            rec["seconds"] = round(time.time() - t0, 1)
            report["runs"].append(rec)
            brief = {k: rec.get(k) for k in ("ok", "train_seconds", "peak_gpu_gb", "error")}
            brief |= {"nll": [rec.get("before", {}).get("forget_nll"), rec.get("after", {}).get("forget_nll"),
                              rec.get("before", {}).get("retain_nll"), rec.get("after", {}).get("retain_nll")],
                      "changed": (rec.get("changed") or {}).get("by_kind")}
            log(f"{method}/{target}: {json.dumps(brief)}")
            (out / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            model = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    report["all_ok"] = all(r["ok"] for r in report["runs"])
    (out / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"all ok: {report['all_ok']}")


if __name__ == "__main__":
    main()
