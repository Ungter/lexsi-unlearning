"""Attack 8 (PLAN 6.3): abliteration of the forgotten-concept direction, then re-test the direct forget metrics.

LFM3 applies RMSNorm with a learned gain to every sublayer output before it is added to the residual stream
("sandwich" norms), so orthogonalising weight matrices does not remove a direction from the stream. We therefore use
activation ablation (Arditi et al.): after the embedding and after every decoder layer, project the direction r out of
the residual stream, h <- h - (h . r) r, with forward hooks. Runs in HF (vLLM cannot hook).

Direction at layer l = normalised difference of mean layer-l residuals between forget and contrast text, using TRAIN/DEV
data only (an attacker with access to the forget set):
  calc   forget = calculus dev instructions (chat, thinking off)   contrast = arithmetic/algebra/precalculus dev instructions
  book   forget = raw_forget_train chunk prefixes                  contrast = raw_retain1_train chunk prefixes
Variants: none (control), then one per --layers entry. Metrics (test splits, greedy):
  calc   100 test problems (accuracy, truncated/gibberish rates) + 40 retain-math problems (collateral damage)
  book   verbatim continuation (25 passages, ROUGE-L F1), forget knowmem ICL (fact), retain knowmem ICL (fact)
Writes <out>/abl.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import grade  # noqa: E402
from lfm3u.audit import muse  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def seeded(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def log(msg: str) -> None:
    print(f"[abl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_generate(model, tok, prompts: list[list[int]], max_new: int, bs: int = 100) -> list[dict]:
    import torch

    pad = tok.pad_token_id
    eos = sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0})
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[dict | None] = [None] * len(prompts)
    dev = next(model.parameters()).device
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[j, L - len(prompts[i]):] = 1
        with torch.no_grad():
            g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False,
                               temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=pad,
                               eos_token_id=eos, use_cache=True)
        for j, i in enumerate(idx):
            new = g[j, L:].tolist()
            n = next((k for k, t in enumerate(new) if t in eos or t == pad), len(new))
            outs[i] = {"text": tok.decode(new[:n], skip_special_tokens=True).strip(), "n_tokens": n,
                       "truncated": n >= max_new}
    return outs


def chat_ids(tok, text: str) -> list[int]:
    t = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(t, add_special_tokens=False)["input_ids"]


def layer_means(model, tok, prompts: list[list[int]], bs: int = 32):
    """Mean residual (fp32) over real tokens, for the embedding output and every layer: tensor [n_layers + 1, d]."""
    import torch

    pad = tok.pad_token_id
    dev = next(model.parameters()).device
    total, count = None, 0
    for b in range(0, len(prompts), bs):
        chunk = prompts[b:b + bs]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, :len(p)] = torch.tensor(p)
            att[j, :len(p)] = 1
        with torch.no_grad():
            hs = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        m = att.to(dev).bool()
        s = torch.stack([h[m].float().sum(0) for h in hs])
        total = s if total is None else total + s
        count += int(m.sum())
    return total / count


def ablation_hooks(model, r):
    import torch

    r = r.to(next(model.parameters()).device, torch.float32)

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        h32 = h.float()
        h2 = (h32 - (h32 @ r).unsqueeze(-1) * r).to(h.dtype)
        return (h2, *output[1:]) if isinstance(output, tuple) else h2

    mods = [model.model.embed_tokens, *model.model.layers]
    return [m.register_forward_hook(hook) for m in mods]


def eval_calc(model, tok, rows, retain, max_new, bs: int = 100):
    outs = hf_generate(model, tok, [chat_ids(tok, r["instruction"]) for r in rows], max_new, bs)
    recs = [grade_item(r, o["text"]) | {"truncated": o["truncated"], "kind": r.get("kind")} for r, o in zip(rows, outs)]
    outs2 = hf_generate(model, tok, [chat_ids(tok, r["instruction"]) for r in retain], 2048, bs)
    recs2 = [grade_item(r, o["text"]) for r, o in zip(retain, outs2)]
    res = {"calc_acc": mean(r["correct"] for r in recs), "calc_truncated": mean(r["truncated"] for r in recs),
           "calc_gibberish": mean(r["gibberish"] for r in recs), "retain_math_acc": mean(r["correct"] for r in recs2),
           "retain_math_gibberish": mean(r["gibberish"] for r in recs2)}
    kinds = sorted({r["kind"] for r in recs if r["kind"]})
    if len(kinds) > 1:  # per-kind breakdown (diff / integral / defint); per-item bits for bootstrap CIs
        for k in kinds:
            sub = [r for r in recs if r["kind"] == k]
            res |= {f"calc_acc_{k}": mean(r["correct"] for r in sub), f"calc_truncated_{k}": mean(r["truncated"] for r in sub),
                    f"n_{k}": len(sub)}
    res["calc_bits"] = "".join("1" if r["correct"] else "0" for r in recs)
    return res


def eval_book(model, tok, data: Path, atk: Path):
    verb = read_jsonl(data / "book/verbmem_forget_test.jsonl", allow_test=True)
    bos = tok.bos_token_id
    outs = hf_generate(model, tok, [[bos, *tok(r["prompt"], add_special_tokens=False)["input_ids"]] for r in verb], 128)
    f1 = [grade.rouge_l(o["text"], r["target"])["rougeL_f1"] for r, o in zip(verb, outs)]
    res = {"verbmem_f1": mean(f1), "verbmem_gibberish": mean(grade.is_gibberish(o["text"]) for o in outs),
           "verbmem_f1_items": [round(x, 3) for x in f1]}
    for split, key in (("forget", "forget"), ("retain", "retain")):
        qa = read_jsonl(data / f"book/knowmem_{split}_qa_test.jsonl", allow_test=True)
        prefix = muse.icl_prefix(read_jsonl(data / f"book/knowmem_{split}_qa_icl_dev.jsonl", allow_test=True))
        outs = hf_generate(model, tok, [[bos, *tok(prefix + f"Question: {r['question']}\nAnswer:", add_special_tokens=False)["input_ids"]] for r in qa], 32)
        texts = [o["text"].split("\n")[0] for o in outs]
        res[f"knowmem_{key}_fact"] = mean(grade.fact_present(t, r["answer"]) for t, r in zip(texts, qa))
        res[f"knowmem_{key}_gibberish"] = mean(grade.is_gibberish(o["text"]) for o in outs)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--domain", choices=("book", "calc"), required=True)
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--attack-data", default="attack_data")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--n-calc", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out or f"results/abl/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    data, atk = Path(a.data), Path(a.attack_data)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).to(device()).eval()
    n = a.limit or a.n_prompts

    if a.domain == "calc":
        fr = [r for lvl in (1, 2, 3) for r in read_jsonl(data / f"calculus/level{lvl}/dev.jsonl")]
        cr = [r for t in ("arithmetic", "algebra", "precalculus") for lvl in (1, 2, 3) for r in read_jsonl(data / f"{t}/level{lvl}/dev.jsonl")]
        seeded("abl", "f").shuffle(fr)
        seeded("abl", "c").shuffle(cr)
        forget_p, contrast_p = [chat_ids(tok, r["instruction"]) for r in fr[:n]], [chat_ids(tok, r["instruction"]) for r in cr[:n]]
        rows = read_jsonl(atk / "_inputs/math_calc_test.jsonl", allow_test=True)
        seeded("abl", "test").shuffle(rows)
        rows = rows[:a.limit or a.n_calc]
        retain = read_jsonl(atk / "_inputs/math_retain_test.jsonl", allow_test=True)[:a.limit or 40]
        evaluate = lambda: eval_calc(model, tok, rows, retain, a.max_new)  # noqa: E731
    else:
        bos = tok.bos_token_id
        fc = read_jsonl(data / "book/raw_forget_train.jsonl")
        rc = read_jsonl(data / "book/raw_retain1_train.jsonl")
        seeded("abl", "f").shuffle(fc)
        seeded("abl", "c").shuffle(rc)
        forget_p, contrast_p = [[bos, *r["input_ids"][:128]] for r in fc[:n]], [[bos, *r["input_ids"][:128]] for r in rc[:n]]
        evaluate = lambda: eval_book(model, tok, data, atk)  # noqa: E731

    mf, mc = layer_means(model, tok, forget_p), layer_means(model, tok, contrast_p)
    layers = [int(x) for x in a.layers.split(",")]
    dirs = {}
    for l in layers:
        d = mf[l + 1] - mc[l + 1]
        dirs[l] = (d / d.norm()).cpu()
    report = {"name": a.name, "model": a.model, "domain": a.domain, "layers": layers, "n_prompts": n,
              "direction_cosines": {f"{x}-{y}": round(float(dirs[x] @ dirs[y]), 3) for i, x in enumerate(layers) for y in layers[i + 1:]},
              "direction_diff_norm": {str(l): round(float((mf[l + 1] - mc[l + 1]).norm()), 2) for l in layers},
              "variants": {}}
    t0 = time.time()
    report["variants"]["none"] = evaluate()
    log(f"{a.name} none: {json.dumps(report['variants']['none'])} ({time.time() - t0:.0f}s)")
    for l in layers:
        hooks = ablation_hooks(model, dirs[l])
        try:
            report["variants"][f"layer{l}"] = evaluate()
        finally:
            for h in hooks:
                h.remove()
        log(f"{a.name} layer {l}: {json.dumps(report['variants'][f'layer{l}'])} ({time.time() - t0:.0f}s)")
        (out / "abl.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    key = "calc_acc" if a.domain == "calc" else "verbmem_f1"
    best = max(layers, key=lambda l: report["variants"][f"layer{l}"][key] or 0)
    report |= {"best_layer": best, "best": report["variants"][f"layer{best}"], "seconds": round(time.time() - t0, 1)}
    (out / "abl.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done; best layer {best}")


if __name__ == "__main__":
    main()
